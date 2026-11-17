#!/usr/bin/env python3
"""
Phase 2 + glue: FastAPI controller for the Glass Box system.

This server:
  1. Spawns and supervises `llama-server` (the INT4 1.5B inference backend).
  2. Spawns and supervises the viz model WebSocket server (Phase 3).
  3. Exposes a session concept with a pinned prefix:
       POST /session         -> returns {session_id, n_cached}
       POST /generate        -> body: {session_id, message} -> streams tokens
       DELETE /session/{id}  -> drops the slot
  4. Tracks per-request: tokens served from cache vs freshly computed,
     latency, peak memory.
  5. Fans out per-token events to /ws/dashboard (the frontend).

The KV cache is managed by llama-server's native slot system. We pin one
slot per session, set its initial prefix with /apply-template + a series
of /tokenize calls + slot.actions = ["save"], and on subsequent
/generate calls in that session we re-load the saved slot state. That's
llama.cpp's idiomatic way to keep a session-pinned KV cache.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import psutil
import websockets
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = Path(__file__).resolve().parent.parent
LLAMA_CPP_BIN = REPO_ROOT / "llama.cpp" / "build" / "bin" / "llama-server"
FRONTEND_DIR = REPO_ROOT / "frontend"


# ------------------------------------------------------------------
# Session state
# ------------------------------------------------------------------

@dataclass
class Session:
    """A pinned-prefix session on llama-server.

    We track the full conversation history (system + alternating user/assistant
    turns) so we can re-send it on every request. This is essential for the
    KV-cache slot to match the prompt and actually reuse cached keys/values.
    If we only sent the new user message, llama-server would see a prompt
    that doesn't match the slot's existing state and could generate
    runaway tokens from the wrong position.
    """
    session_id: str
    slot_id: int
    prefix_text: str           # the system prompt
    prefix_tokens: list[int]   # tokenized prefix (system message rendered)
    created_at: float
    last_used: float
    # Conversation history: list of {"role": ..., "content": ...}
    history: list[dict] = field(default_factory=list)
    # Cache accounting — populated as the session is used.
    total_cached_tokens: int = 0
    total_fresh_tokens: int = 0
    n_requests: int = 0
    # Whether the viz model has been pre-filled with this session's
    # growing prompt yet. We re-prefill only when the prompt changes
    # (i.e. on a new /generate), and only the first time for a given
    # session+message pair — subsequent tokens are `step`s.
    viz_prefilled_for: str = ""  # the message we last pre-filled for

    @property
    def cache_hit_rate(self) -> float:
        total = self.total_cached_tokens + self.total_fresh_tokens
        return self.total_cached_tokens / total if total > 0 else 0.0


class SessionManager:
    """Owns the set of active sessions; round-robins slot ids."""

    def __init__(self, max_slots: int):
        self.max_slots = max_slots
        self._sessions: dict[str, Session] = {}
        self._used_slots: set[int] = set()

    def create(self, prefix: str, prefix_tokens: list[int]) -> Session:
        if len(self._used_slots) >= self.max_slots:
            raise RuntimeError(f"No free slots (max {self.max_slots})")
        slot_id = next(i for i in range(self.max_slots) if i not in self._used_slots)
        sid = uuid.uuid4().hex[:12]
        sess = Session(
            session_id=sid,
            slot_id=slot_id,
            prefix_text=prefix,
            prefix_tokens=prefix_tokens,
            created_at=time.time(),
            last_used=time.time(),
        )
        self._sessions[sid] = sess
        self._used_slots.add(slot_id)
        return sess

    def get(self, sid: str) -> Session | None:
        return self._sessions.get(sid)

    def delete(self, sid: str) -> bool:
        sess = self._sessions.pop(sid, None)
        if sess is None:
            return False
        self._used_slots.discard(sess.slot_id)
        return True

    def list(self) -> list[Session]:
        return list(self._sessions.values())


# ------------------------------------------------------------------
# Memory tracker — measures peak RSS, GPU mem, swap across the run.
# ------------------------------------------------------------------

class MemoryTracker:
    def __init__(self):
        self.proc = psutil.Process(os.getpid())
        self.peak_rss_mb = 0.0
        self.peak_gpu_mb = 0.0

    def sample(self) -> dict:
        rss_mb = self.proc.memory_info().rss / 1e6
        self.peak_rss_mb = max(self.peak_rss_mb, rss_mb)
        gpu_mb = 0.0
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem = pynvml.nvmlDeviceGetMemoryInfo(h)
            gpu_mb = mem.used / 1e6
            pynvml.nvmlShutdown()
        except Exception:
            pass
        self.peak_gpu_mb = max(self.peak_gpu_mb, gpu_mb)
        return {
            "rss_mb": rss_mb,
            "gpu_mb": gpu_mb,
            "total_mb": psutil.virtual_memory().total / 1e6,
            "available_mb": psutil.virtual_memory().available / 1e6,
        }


# ------------------------------------------------------------------
# llama-server process supervisor
# ------------------------------------------------------------------

class LlamaServerSupervisor:
    """Starts llama-server and waits until its HTTP API is responsive."""

    def __init__(self, model_path: Path, host: str, port: int, n_gpu_layers: int, ctx: int,
                 llama_bin: Path | None = None, extra_args: list[str] | None = None):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.n_gpu_layers = n_gpu_layers
        self.ctx = ctx
        self.llama_bin = llama_bin or LLAMA_CPP_BIN
        self.extra_args = extra_args or []
        self.proc: subprocess.Popen | None = None
        self._log_file: Path | None = None

    def start(self) -> None:
        if not self.llama_bin.exists():
            sys.exit(f"missing {self.llama_bin}; build llama.cpp first or pass --llama-bin")
        if not self.model_path.exists():
            sys.exit(f"missing model {self.model_path}; run quantize first")

        log_path = REPO_ROOT / "logs" / "llama-server.log"
        log_path.parent.mkdir(exist_ok=True)
        self._log_file = open(log_path, "w")

        cmd = [
            str(self.llama_bin),
            "-m", str(self.model_path),
            "--host", self.host,
            "--port", str(self.port),
            "-ngl", str(self.n_gpu_layers),
            "-c", str(self.ctx),
            "--cont-batching",
            # expose metrics so we can read cache hits
            "--metrics",
        ] + list(self.extra_args)
        print(f"[ctrl] spawning: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )
        # Wait for /health
        base = f"http://{self.host}:{self.port}"
        for i in range(120):
            try:
                r = httpx.get(f"{base}/health", timeout=1.0)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    print(f"[ctrl] llama-server healthy after {i*0.5:.1f}s")
                    return
            except Exception:
                pass
            time.sleep(0.5)
        self.stop()
        sys.exit("[ctrl] llama-server failed to become healthy in 60s; see logs/llama-server.log")

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        if self._log_file:
            self._log_file.close()


# ------------------------------------------------------------------
# Viz model process supervisor (Phase 3)
# ------------------------------------------------------------------

class VizModelSupervisor:
    """Spawns and supervises `python3 src/viz_model.py` and waits until
    its /health endpoint responds. Mirrors LlamaServerSupervisor, plus
    runs a background watchdog that auto-restarts the process if it
    dies (OOM kill, crash, anything). The first healthy start still
    blocks; restarts happen silently in the background so the demo
    doesn't notice."""

    def __init__(self, host: str, port: int, clusters_path: Path, model: str,
                 viz_python: str | None = None):
        self.host = host
        self.port = port
        self.clusters_path = clusters_path
        self.model = model
        self.viz_python = viz_python or sys.executable
        self.proc: subprocess.Popen | None = None
        self._log_file: Path | None = None
        self._watchdog_stop = threading.Event()
        self._watchdog_thread: threading.Thread | None = None

    def _spawn(self) -> None:
        cmd = [
            self.viz_python, str(REPO_ROOT / "src" / "viz_model.py"),
            "--model", self.model,
            "--clusters", str(self.clusters_path),
            "--port", str(self.port),
        ]
        # Reopen the log in append mode for restarts so we keep history.
        if self._log_file is None:
            self._log_file = open(REPO_ROOT / "logs" / "viz.log", "a")
        print(f"[ctrl] spawning: {' '.join(cmd)}")
        self.proc = subprocess.Popen(
            cmd,
            stdout=self._log_file,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

    def _wait_healthy(self, deadline_s: float) -> bool:
        base = f"http://{self.host}:{self.port}"
        while time.time() < deadline_s:
            try:
                r = httpx.get(f"{base}/health", timeout=1.0)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return True
            except Exception:
                pass
            if self.proc.poll() is not None:
                return False
            time.sleep(0.5)
        return False

    def start(self) -> None:
        if not self.clusters_path.exists():
            sys.exit(f"missing {self.clusters_path}; run neuron_clusters.py first")
        viz_script = REPO_ROOT / "src" / "viz_model.py"
        if not viz_script.exists():
            sys.exit(f"missing {viz_script}")

        # Initial spawn + wait for first healthy.
        self._spawn()
        if not self._wait_healthy(time.time() + 90):
            self.stop()
            sys.exit("[ctrl] viz model failed to become healthy in 90s; see logs/viz.log")
        print("[ctrl] viz model healthy")

        # Background watchdog: respawn if the process dies, with backoff
        # to avoid tight restart loops if something is structurally wrong.
        self._watchdog_stop.clear()
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop, daemon=True, name="viz-watchdog"
        )
        self._watchdog_thread.start()

    def _watchdog_loop(self) -> None:
        backoff = 2.0
        while not self._watchdog_stop.is_set():
            # Check every 2s whether the process is still alive.
            self._watchdog_stop.wait(2.0)
            if self._watchdog_stop.is_set():
                return
            if self.proc is None:
                continue
            rc = self.proc.poll()
            if rc is None:
                backoff = 2.0  # reset on healthy
                continue
            # Process died. Log and respawn.
            print(f"[ctrl] viz model exited with rc={rc}; restarting in {backoff:.1f}s", flush=True)
            self._watchdog_stop.wait(backoff)
            backoff = min(backoff * 2, 30.0)
            if self._watchdog_stop.is_set():
                return
            try:
                self._spawn()
                # Give it 60s to come back. If it doesn't, the next
                # iteration of this loop will see poll() != None again
                # and try again.
                self._wait_healthy(time.time() + 60)
            except Exception as e:
                print(f"[ctrl] viz respawn failed: {e}", flush=True)

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            try:
                os.killpg(os.getpgid(self.proc.pid), signal.SIGTERM)
                self.proc.wait(timeout=5)
            except Exception:
                try:
                    os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
                except Exception:
                    pass
        if self._log_file:
            self._log_file.close()


# ------------------------------------------------------------------
# The FastAPI app itself
# ------------------------------------------------------------------

def build_app(args) -> FastAPI:
    app = FastAPI(title="Glass Box controller")

    sessions = SessionManager(max_slots=args.max_sessions)
    mem = MemoryTracker()
    dashboards: set[WebSocket] = set()

    sup = LlamaServerSupervisor(
        model_path=Path(args.model),
        host="127.0.0.1",
        port=args.llama_port,
        n_gpu_layers=args.n_gpu_layers,
        ctx=args.ctx,
        llama_bin=Path(args.llama_bin),
        extra_args=shlex.split(args.llama_extra_args or ""),
    )
    sup.start()
    LLAMA_BASE = f"http://127.0.0.1:{args.llama_port}"

    # Phase 3: spawn the viz model so the controller can forward
    # per-token events to it and the frontend can stream activations.
    viz_sup = VizModelSupervisor(
        host="127.0.0.1",
        port=args.viz_port,
        clusters_path=Path(args.viz_clusters),
        model=args.viz_model,
        viz_python=args.viz_python,
    )
    viz_sup.start()
    VIZ_WS_URL = f"ws://127.0.0.1:{args.viz_port}/ws/viz"

    # Persistent internal WebSocket client to the viz model (Bug 3).
    # Opened once in the startup hook, reused for every /generate call,
    # closed in the shutdown hook. If it ever disconnects we lazily
    # reconnect on the next call.
    class VizClient:
        def __init__(self):
            self.ws = None
            self.lock = asyncio.Lock()

        async def ensure(self):
            async with self.lock:
                if self.ws is None:
                    self.ws = await websockets.connect(VIZ_WS_URL, open_timeout=10)
                    return
                # If the socket is closed, drop and reopen.
                if self.ws.state.name != "OPEN":
                    try:
                        await self.ws.close()
                    except Exception:
                        pass
                    self.ws = None
                    self.ws = await websockets.connect(VIZ_WS_URL, open_timeout=10)

        async def call(self, payload: dict, timeout: float = 30.0) -> dict:
            """Send a JSON message and await one response. Reconnects on failure."""
            for attempt in range(2):
                try:
                    await self.ensure()
                    await self.ws.send(json.dumps(payload))
                    raw = await asyncio.wait_for(self.ws.recv(), timeout=timeout)
                    return json.loads(raw)
                except Exception:
                    # Drop the socket and retry once.
                    try:
                        if self.ws is not None:
                            await self.ws.close()
                    except Exception:
                        pass
                    self.ws = None
                    if attempt == 1:
                        raise
            raise RuntimeError("unreachable")

        async def close(self):
            if self.ws is not None:
                try:
                    await self.ws.close()
                except Exception:
                    pass
                self.ws = None

    viz_client = VizClient()

    @app.on_event("startup")
    async def _viz_open():
        await viz_client.ensure()
        print(f"[ctrl] viz WebSocket connected: {VIZ_WS_URL}")

    async def broadcast(payload: dict) -> None:
        dead = []
        for ws in list(dashboards):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for d in dead:
            dashboards.discard(d)

    def tokenize_prefix(prefix: str) -> list[int]:
        """Tokenize via llama-server's /tokenize endpoint. Returns plain int list."""
        r = httpx.post(f"{LLAMA_BASE}/tokenize", json={"content": prefix, "add_special": True}, timeout=30)
        r.raise_for_status()
        # llama.cpp returns {"tokens": [...]} where tokens may be objects or ints depending on version.
        toks = r.json()["tokens"]
        if toks and isinstance(toks[0], dict):
            return [t["id"] for t in toks]
        return list(toks)

    def slot_state(slot_id: int) -> dict | None:
        """Return the current state of a slot, or None if not found."""
        r = httpx.get(f"{LLAMA_BASE}/slots", timeout=5)
        r.raise_for_status()
        for s in r.json():
            if s.get("id") == slot_id:
                return s
        return None

    def erase_slot(slot_id: int) -> None:
        """Erase the slot's KV cache. Slot remains allocated."""
        r = httpx.post(f"{LLAMA_BASE}/slots/{slot_id}?action=erase", timeout=10)
        r.raise_for_status()

    @app.on_event("shutdown")
    async def _shutdown():
        await viz_client.close()
        sup.stop()
        viz_sup.stop()

    # -- API endpoints --

    @app.post("/session")
    async def create_session(payload: dict):
        prefix = payload.get("prefix", "")
        if not prefix:
            return JSONResponse({"error": "prefix is required"}, status_code=400)
        prefix_tokens = tokenize_prefix(prefix)

        sess = sessions.create(prefix, prefix_tokens)

        # Seed the slot with the EXACT prompt shape the first /generate
        # will use. The chat template wraps system+user into a specific
        # token sequence; if we don't seed with that exact sequence,
        # the first generate gets 0 cache hits because the slot's KV
        # doesn't match.
        seed_messages = [
            {"role": "system", "content": prefix},
            {"role": "user", "content": ""},  # placeholder, ignored by n_predict=1
        ]
        async with httpx.AsyncClient(timeout=120) as cx:
            r = await cx.post(
                f"{LLAMA_BASE}/v1/chat/completions",
                json={
                    "model": "any",
                    "messages": seed_messages,
                    "max_tokens": 1,
                    "slot_id": sess.slot_id,
                    "stream": False,
                    "cache_prompt": True,
                },
            )
        r.raise_for_status()
        # The slot's KV cache now holds the system+user template. The
        # slot stays alive across subsequent requests in this session.
        ud = r.json().get("usage", {})
        st = slot_state(sess.slot_id) or {}

        return {
            "session_id": sess.session_id,
            "slot_id": sess.slot_id,
            "prefix_tokens": len(prefix_tokens),
            "cached_after_seed": ud.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            "prompt_tokens_seeded": ud.get("prompt_tokens", 0),
        }

    @app.delete("/session/{sid}")
    async def delete_session(sid: str):
        sess = sessions.get(sid)
        if not sess:
            return JSONResponse({"error": "unknown session"}, status_code=404)
        try:
            erase_slot(sess.slot_id)
        except Exception:
            pass
        sessions.delete(sid)
        return {"ok": True}

    @app.get("/sessions")
    async def list_sessions():
        out = []
        for s in sessions.list():
            out.append({
                "session_id": s.session_id,
                "slot_id": s.slot_id,
                "prefix_tokens": len(s.prefix_tokens),
                "n_requests": s.n_requests,
                "total_cached": s.total_cached_tokens,
                "total_fresh": s.total_fresh_tokens,
                "cache_hit_rate": round(s.cache_hit_rate, 4),
                "created_at": s.created_at,
                "last_used": s.last_used,
            })
        return {"sessions": out}

    @app.post("/generate")
    async def generate(payload: dict):
        """Generate within a session.

        We use llama-server's OAI-compatible /v1/chat/completions endpoint
        with slot_id pinning. This makes llama-server keep the slot's KV
        cache resident across requests in the same slot_id, and the
        response's `usage.prompt_tokens_details.cached_tokens` reports
        how many of the prompt tokens were served from cache.
        """
        sid = payload.get("session_id")
        message = payload.get("message", "")
        max_tokens = int(payload.get("max_tokens", 64))
        if not sid or not message:
            return JSONResponse({"error": "session_id and message required"}, status_code=400)
        sess = sessions.get(sid)
        if not sess:
            return JSONResponse({"error": "unknown session"}, status_code=404)

        sess.n_requests += 1
        t0 = time.time()

        # Build the OAI-style messages list with FULL conversation
        # history. The slot's KV cache only matches when the rendered
        # prompt exactly equals what it had before. Sending only the
        # latest user message would cause llama-server to evaluate the
        # whole chat-template prefix again (cache miss on prefix) AND
        # could lead to runaway generation from a mismatched position.
        oai_messages = [{"role": "system", "content": sess.prefix_text}]
        oai_messages.extend(sess.history)
        oai_messages.append({"role": "user", "content": message})

        # Render the OAI messages as plain text for the viz model. The
        # viz model (0.5B) is just used to visualize activations on the
        # same text the served model is processing — it doesn't need to
        # match the served model's exact chat template, just see a
        # reasonable approximation of the prompt content.
        viz_prompt_text = "\n".join(
            m["content"] for m in oai_messages if m.get("role") in ("system", "user")
        )

        # If this is a new user turn for this session (or first call),
        # prefill the viz model with the prompt. The viz model is a
        # different model with a different tokenizer than the served
        # one (Qwen2.5-0.5B vs 1.5B), so the per-step token ids come
        # from the viz model's own tokenization of the growing text.
        # Design choice: we don't try to keep the two token streams in
        # lock-step — the visualization is a proxy for "what the
        # activations look like on this text", not a bit-exact replay
        # of the served model's compute. See viz_model.py module
        # docstring for the rationale.
        if sess.viz_prefilled_for != message:
            try:
                prefill_resp = await viz_client.call(
                    {"kind": "prefill", "prompt": viz_prompt_text},
                    timeout=60.0,
                )
                await broadcast({
                    "kind": "prefill_done",
                    "session_id": sid,
                    "message": message,
                    **prefill_resp,
                })
                sess.viz_prefilled_for = message
            except Exception as e:
                # Don't fail the whole /generate if the viz model is
                # unreachable — the served model's output is the
                # primary deliverable.
                print(f"[ctrl] viz prefill failed: {e}", flush=True)

        # Thinking models (Qwen3.5-style) stream reasoning first via
        # delta.reasoning_content, then the answer via delta.content.
        # Allow the caller to toggle thinking per request; fall back to a
        # startup default passed as --chat-template-kwargs.
        chat_template_kwargs = payload.get("chat_template_kwargs")
        if chat_template_kwargs is None and args.chat_template_kwargs:
            chat_template_kwargs = json.loads(args.chat_template_kwargs)

        async with httpx.AsyncClient(timeout=300) as cx:
            # Stream so the dashboard sees tokens live. Ask for usage
            # chunk via stream_options.include_usage — the server emits
            # it as a final chunk with no content, only usage.
            full_text = []
            reasoning_text = []
            tokens: list[dict] = []
            cached_so_far = 0
            prompt_total = 0
            async with cx.stream(
                "POST",
                f"{LLAMA_BASE}/v1/chat/completions",
                json={
                    **({"chat_template_kwargs": chat_template_kwargs}
                       if chat_template_kwargs is not None else {}),
                    "model": "any",
                    "messages": oai_messages,
                    "max_tokens": max_tokens,
                    "slot_id": sess.slot_id,
                    "stream": True,
                    "stream_options": {"include_usage": True},
                    "temperature": float(payload.get("temperature", args.temperature)),
                    "top_k": int(payload.get("top_k", args.top_k)),
                    "top_p": float(payload.get("top_p", args.top_p)),
                    "cache_prompt": True,
                    # Repeat penalty helps avoid degenerate loops on
                    # quantized small models in interactive sessions.
                    "repeat_penalty": float(payload.get("repeat_penalty", args.repeat_penalty)),
                },
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    data = line[6:]
                    if data == "[DONE]":
                        break
                    obj = json.loads(data)
                    # Usage chunk has empty choices
                    if obj.get("usage"):
                        ud = obj["usage"]
                        prompt_total = ud.get("prompt_tokens", 0)
                        cached_so_far = (
                            ud.get("prompt_tokens_details", {}).get("cached_tokens", 0)
                        )
                        continue
                    # Normal delta chunk. Reasoning models: reasoning_content
                    # arrives before content — stream both, flagged, so the
                    # dashboard and the viz heatmap light up while it thinks.
                    try:
                        delta = obj["choices"][0].get("delta", {})
                        tok = delta.get("content") or ""
                        reason = delta.get("reasoning_content") or ""
                    except (KeyError, IndexError):
                        tok, reason = "", ""
                    piece = tok or reason
                    if not piece:
                        continue
                    is_reasoning = not tok
                    if is_reasoning:
                        reasoning_text.append(reason)
                    else:
                        full_text.append(tok)
                    tokens.append({
                        "text": piece,
                        "reasoning": is_reasoning,
                        "cached": False,
                        "t_ms": int((time.time() - t0) * 1000),
                    })
                    await broadcast({
                        "kind": "token",
                        "session_id": sid,
                        "token": piece,
                        "reasoning": is_reasoning,
                        "index": len(tokens),
                        "t_ms": tokens[-1]["t_ms"],
                    })

                    # Bug 3: forward each generated token to the viz
                    # model so the dashboard's heatmap can show per-token
                    # activation deltas. The viz model has its own
                    # tokenizer; we ask it to tokenize the full growing
                    # text and feed it the last token id of the new
                    # chunk. See the design comment at the prefill site
                    # above for why we don't try to share token ids
                    # with the served model.
                    try:
                        cum_text = message + "".join(reasoning_text) + "".join(full_text)
                        tok_resp = await cx.post(
                            f"http://127.0.0.1:{args.viz_port}/tokenize",
                            json={"text": cum_text},
                            timeout=10,
                        )
                        tok_resp.raise_for_status()
                        ids = tok_resp.json().get("ids", [])
                        if ids:
                            step_resp = await viz_client.call(
                                {"kind": "step", "token_id": ids[-1]},
                                timeout=10.0,
                            )
                            await broadcast({
                                "kind": "step_done",
                                "session_id": sid,
                                "index": len(tokens),
                                **step_resp,
                            })
                    except Exception as e:
                        # viz failures must not block the served model
                        # output from reaching the user.
                        print(f"[ctrl] viz step failed: {e}", flush=True)

        elapsed = time.time() - t0
        sess.total_cached_tokens += cached_so_far
        sess.total_fresh_tokens += len(tokens)
        sess.last_used = time.time()
        # Append this turn to history so subsequent calls in this
        # session re-send the full conversation and the KV cache hits.
        sess.history.append({"role": "user", "content": message})
        sess.history.append({"role": "assistant", "content": "".join(full_text)})

        result = {
            "session_id": sid,
            "slot_id": sess.slot_id,
            "tokens": tokens,
            "text": "".join(full_text),
            "n_generated": len(tokens),
            "reasoning_text": "".join(reasoning_text),
            "n_cached_total": cached_so_far,
            "n_prompt_total": prompt_total,
            "elapsed_s": elapsed,
            "tokens_per_s": len(tokens) / elapsed if elapsed > 0 else 0,
            "memory": mem.sample(),
        }
        await broadcast({"kind": "generate_done", **result})
        return result

    @app.get("/health")
    async def health():
        return {
            "controller": "ok",
            "llama_server": "ok",
            "memory": mem.sample(),
            "n_sessions": len(sessions.list()),
        }

    @app.get("/v1/models")
    async def models():
        return {"models": [{
            "id": Path(args.model).stem,
            "backend": "llama.cpp",
            "path": str(args.model),
        }]}

    # -- Dashboard WebSocket --

    @app.get("/config")
    async def config():
        """Tells the frontend where the viz model lives so it can open
        its own WebSocket to the right host:port (Bug 2)."""
        return {
            "viz_ws_url": f"ws://{args.controller_host}:{args.viz_port}/ws/viz",
            "served_model": Path(args.model).name,
        }

    @app.websocket("/ws/dashboard")
    async def ws_dashboard(ws: WebSocket):
        await ws.accept()
        dashboards.add(ws)
        try:
            await ws.send_json({"kind": "hello", "memory": mem.sample()})
            while True:
                # We don't expect dashboard to send much; just keep alive.
                msg = await ws.receive_text()
                if msg == "ping":
                    await ws.send_json({"kind": "pong"})
        except WebSocketDisconnect:
            pass
        finally:
            dashboards.discard(ws)

    # -- Static frontend --
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    async def root():
        return FileResponse(str(FRONTEND_DIR / "index.html"))

    return app


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="Path to GGUF INT4 model")
    p.add_argument("--llama-port", type=int, default=8080)
    p.add_argument("--controller-port", type=int, default=8000)
    p.add_argument("--n-gpu-layers", type=int, default=99, help="Offload all layers to GPU")
    p.add_argument("--llama-bin", default=str(LLAMA_CPP_BIN),
                   help="Path to a llama-server binary (use an existing CUDA build)")
    p.add_argument("--llama-extra-args", default="",
                   help="Extra CLI flags appended to the llama-server launch (shell-quoted)")
    p.add_argument("--viz-python", default=sys.executable,
                   help="Python interpreter for the viz model process (e.g. CUDA-torch venv)")
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repeat-penalty", type=float, default=1.1)
    p.add_argument("--chat-template-kwargs", default="",
                   help='Default chat template kwargs as JSON, e.g. {"enable_thinking": false}')
    p.add_argument("--ctx", type=int, default=2048)
    p.add_argument("--max-sessions", type=int, default=4)
    p.add_argument("--controller-host", default="localhost",
                   help="Hostname the dashboard runs on (used to build the viz ws url for the frontend)")
    p.add_argument("--viz-port", type=int, default=8765,
                   help="Port the supervised viz_model.py listens on")
    p.add_argument("--viz-model", default="Qwen/Qwen2.5-0.5B-Instruct",
                   help="HuggingFace model id for the viz model")
    p.add_argument("--viz-clusters", default=str(REPO_ROOT / "models" / "neuron_clusters.json"),
                   help="Path to the precomputed PCA clusters file")
    args = p.parse_args()

    app = build_app(args)
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.controller_port, log_level="info")


if __name__ == "__main__":
    main()
