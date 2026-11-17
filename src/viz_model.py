#!/usr/bin/env python3
"""
Phase 3: Live activation visualization model.

Loads Qwen2.5-0.5B-Instruct in FP16 via transformers, attaches forward hooks
to every decoder layer's residual stream, and exposes a server that streams
per-token per-layer activation projections over a WebSocket.

Why a *second, smaller* model and not the 1.5B served model?

  llama.cpp (which runs the 1.5B INT4 server) is a C++ black box — there is
  no Python hook API. To get per-layer activations we need a PyTorch model
  with real forward hooks. Loading a 1.5B FP16 model *in addition to* the
  INT4 1.5B server would overflow the 8GB Jetson.

  So we run a 0.5B FP16 model on the same prompts/tokens as the INT4
  server. The visualization shows what a comparable model "sees" on the
  same text. This is honest about the tradeoff — the point is to make the
  *cache hit/fresh* distinction visible against a layer-grid of activations,
  and 0.5B is plenty of resolution for that.

Memory: 0.5B in FP16 = ~1 GB weights + ~200 MB activations + ~100 MB KV
cache for 512-token sequences.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_VIZ_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_CLUSTERS_PATH = REPO_ROOT / "models" / "neuron_clusters.json"


def _get_decoder_blocks(model) -> torch.nn.ModuleList:
    # Dense LMs: model.model.layers (Llama/Qwen-style) or transformer.h (GPT-2-style).
    # Multimodal wrappers (e.g. Qwen3.5 ForConditionalGeneration): the text decoder
    # can sit at model.model.language_model.layers (post .to(device) restructure) or
    # behind a top-level model.language_model shim — search both forms.
    cands = [model]
    for outer in ("model", "transformer", "language_model"):
        if hasattr(model, outer):
            cands.append(getattr(model, outer))
    for cand in cands:
        for mid in ("language_model", "text_model"):
            if hasattr(cand, mid):
                inner = getattr(cand, mid)
                for a in ("layers", "h", "blocks"):
                    if hasattr(inner, a):
                        return getattr(inner, a)
        for a in ("layers", "h", "blocks"):
            if hasattr(cand, a):
                return getattr(cand, a)
    raise RuntimeError(f"Could not locate decoder blocks on {type(model).__name__}")


def load_clusters(path: Path) -> tuple[np.ndarray, dict]:
    """Load the saved PCA bases."""
    with open(path, "rb") as f:
        head_line = f.readline()
        meta = json.loads(head_line)
        bases = np.frombuffer(f.read(), dtype=np.float16).reshape(
            meta["num_layers"], meta["hidden_dim"], meta["num_clusters"]
        ).astype(np.float32)
    return bases, meta


class VizRunner:
    """Holds the FP16 viz model. Computes activations per generation step."""

    def __init__(self, model_id: str, clusters_path: Path):
        print(f"[viz] loading {model_id} on cuda (FP16)...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        cfg = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        model_cls = (
            AutoModelForImageTextToText
            if getattr(cfg, "vision_config", None) is not None
            else AutoModelForCausalLM
        )
        print(f"[viz] {type(cfg).__name__} -> {model_cls.__name__}")
        self.model = model_cls.from_pretrained(
            model_id,
            dtype=torch.float16,
            trust_remote_code=True,
        ).to("cuda").eval()
        print(f"[viz] model loaded; {torch.cuda.memory_allocated()/1e9:.2f} GB on GPU")

        self.bases, self.cluster_meta = load_clusters(clusters_path)
        print(f"[viz] loaded clusters: {self.cluster_meta['num_layers']} layers x "
              f"{self.cluster_meta['num_clusters']} clusters, "
              f"hidden_dim={self.cluster_meta['hidden_dim']}")

        # Pre-allocate GPU tensor for the projection bases so per-step compute is fast.
        self.bases_t = torch.from_numpy(self.bases).to("cuda")  # [L, H, K]

        # Hook state: per layer, last forward's [seq, hidden] activations.
        self._acts: dict[int, torch.Tensor] = {}
        # Generation cache: per-step activations are computed WITH the full prefix
        # context (KV for attention layers, recurrent state for linear-attention /
        # GatedDeltaNet hybrid layers). Upstream fed each streamed token through a
        # context-free forward; with hybrid attention that misrepresents the layers.
        self._past = None
        self._register_hooks()

    def _register_hooks(self) -> None:
        blocks = _get_decoder_blocks(self.model)
        for li, block in enumerate(blocks):
            block.register_forward_hook(self._make_hook(li))

    def _make_hook(self, layer_idx: int):
        def hook(module, inputs, outputs):
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            # [batch, seq, hidden] -> take batch 0 (we generate one at a time)
            self._acts[layer_idx] = hidden[0].detach()
        return hook

    @torch.no_grad()
    def prefill(self, prompt: str) -> tuple[list[int], int]:
        """Run the prompt through the model, return (token_ids, n_tokens).

        After this call, self._acts[li] holds the activations for the
        *last* prompt token across all layers. We also keep the full
        sequence activations by re-running hooks with a different mode.
        """
        # Reset act cache
        self._full_acts: dict[int, list[torch.Tensor]] = {}

        # Temporarily replace hooks to capture all tokens, not just last.
        blocks = _get_decoder_blocks(self.model)
        handles = []
        for li, block in enumerate(blocks):
            def make_full_hook(idx):
                def h(m, i, o):
                    h = o[0] if isinstance(o, tuple) else o
                    self._full_acts.setdefault(idx, []).append(h[0].detach().clone())
                return h
            handles.append(block.register_forward_hook(make_full_hook(li)))

        self._past = None  # new prompt -> fresh cache
        try:
            ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
            out = self.model(ids, use_cache=True)
            self._past = out.past_key_values
            n = ids.shape[1]
        finally:
            for h in handles:
                h.remove()

        # Restore simple hooks for generation steps
        self._register_hooks()
        return ids[0].tolist(), n

    @torch.no_grad()
    def step(self, last_token_id: int) -> tuple[np.ndarray, np.ndarray]:
        """Forward one token. Return (clusters [L, K], tokens [L, K]) — actually
        returns the cluster activations per layer for the *new* token position.

        We project the last-token hidden state through the precomputed PCA
        bases, per layer. Output shape: [num_layers, num_clusters].
        """
        x = torch.tensor([[last_token_id]], device="cuda")
        out = self.model(x, past_key_values=self._past, use_cache=True)
        self._past = out.past_key_values
        # self._acts[li] is now [1, hidden] for each layer — the new token.
        rows = []
        for li in range(self.cluster_meta["num_layers"]):
            act = self._acts[li]  # [1, hidden] in fp16
            # Project: cast to fp32 to match bases_t dtype, then matmul.
            proj = act.float() @ self.bases_t[li]
            rows.append(proj[0].cpu().numpy())
        return np.stack(rows, axis=0)  # [L, K]

    def full_prefill_clusters(self) -> np.ndarray:
        """Return prefill activations projected to clusters: [L, T, K]."""
        L = self.cluster_meta["num_layers"]
        K = self.cluster_meta["num_clusters"]
        # The full hook stores one tensor per forward call per layer.
        # Each tensor is [seq, hidden] (batch dim stripped). Stack along
        # batch axis to get [n_forwards, seq, hidden]. We then flatten
        # forwards×seq into a single T axis for the projection.
        # But typically there's exactly 1 forward for a prefill, so the
        # resulting shape is [seq, hidden].
        out = None
        for li in range(L):
            acts = torch.stack(self._full_acts[li], dim=0)  # [n_fwd, seq, hidden]
            # Reshape to [n_fwd*seq, hidden] — flatten the per-forward
            # and per-token axes into one axis.
            n_fwd, seq, hidden = acts.shape
            acts_flat = acts.reshape(n_fwd * seq, hidden)
            T = acts_flat.shape[0]
            # Match dtype — activations are fp16 (model loaded in fp16),
            # bases_t is fp32. Cast activations to fp32 for the matmul.
            proj = (acts_flat.float() @ self.bases_t[li]).cpu().numpy()  # [T, K]
            if out is None:
                out = np.zeros((L, T, K), dtype=np.float32)
            out[li, :T] = proj
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_VIZ_MODEL)
    parser.add_argument("--clusters", default=str(DEFAULT_CLUSTERS_PATH))
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    runner = VizRunner(args.model, Path(args.clusters))

    app = FastAPI()

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.post("/tokenize")
    async def tokenize(payload: dict):
        """Return the viz model's tokenization of an arbitrary text. The
        controller uses this to map the served model's growing assistant
        text into the viz model's own token id space (Bug 3)."""
        text = payload.get("text", "")
        ids = runner.tokenizer(text, add_special_tokens=False).input_ids
        return {"ids": [int(i) for i in ids]}

    @app.websocket("/ws/viz")
    async def ws_viz(ws: WebSocket):
        await ws.accept()
        try:
            while True:
                msg = await ws.receive_json()
                kind = msg.get("kind")
                if kind == "prefill":
                    prompt = msg["prompt"]
                    t0 = time.time()
                    ids, n = runner.prefill(prompt)
                    clusters = runner.full_prefill_clusters()
                    await ws.send_json({
                        "kind": "prefill_done",
                        "num_tokens": n,
                        "elapsed_ms": int((time.time() - t0) * 1000),
                        # We send the per-layer-per-token cluster activations
                        # in a compact form. Float32, ~24 layers x ~32 clusters
                        # x ~200 tokens = 600KB-ish, manageable.
                        "clusters": clusters.flatten().tolist(),
                        "shape": list(clusters.shape),
                        "tokens": runner.tokenizer.batch_decode([[i] for i in ids]),
                    })
                elif kind == "step":
                    last_id = msg["token_id"]
                    t0 = time.time()
                    clusters = runner.step(last_id)  # [L, K]
                    await ws.send_json({
                        "kind": "step_done",
                        "clusters": clusters.flatten().tolist(),
                        "shape": list(clusters.shape),
                        "elapsed_ms": int((time.time() - t0) * 1000),
                    })
                elif kind == "ping":
                    await ws.send_json({"kind": "pong"})
        except WebSocketDisconnect:
            pass

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
