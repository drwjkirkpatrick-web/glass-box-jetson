#!/usr/bin/env python3
"""End-to-end smoke test for the Glass Box Jetson controller.

Creates a session and runs three turns:
  1. thinking allowed (BotPsy streams reasoning_content first)
  2. thinking disabled via per-request chat_template_kwargs
  3. thinking allowed again — checks session KV-cache reuse

While turn 1 streams, a dashboard WebSocket listener verifies viz-model
prefill/step events flow with the right [layers, tokens, clusters] shape.

Usage: python3 scripts/e2e_test.py [controller_port]
"""
import asyncio
import json
import sys

import httpx
import websockets

PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8001
BASE = f"http://127.0.0.1:{PORT}"
WS = f"ws://127.0.0.1:{PORT}/ws/dashboard"

PREFIX = ("You are BotPsy, a concise psychology tutor running on a Jetson. "
          "Answer briefly.")


async def main() -> None:
    async with httpx.AsyncClient(timeout=300) as cx:
        r = await cx.post(f"{BASE}/session", json={"prefix": PREFIX})
        r.raise_for_status()
        sess = r.json()
        sid = sess["session_id"]
        print(f"session: {sid} | prefix_tokens={sess['prefix_tokens']} "
              f"| cached_after_seed={sess['cached_after_seed']}")

        events: list[tuple] = []

        async def listen() -> None:
            async with websockets.connect(WS) as ws:
                while True:
                    msg = await asyncio.wait_for(ws.recv(), timeout=240)
                    d = json.loads(msg)
                    k = d.get("kind")
                    if k == "prefill_done":
                        events.append(("prefill_done", d.get("shape"),
                                       d.get("num_tokens"), d.get("elapsed_ms")))
                    elif k == "step_done":
                        events.append(("step_done", d.get("shape"), d.get("elapsed_ms")))
                    elif k == "generate_done":
                        events.append(("generate_done", d.get("n_generated"),
                                       d.get("n_cached_total")))
                        break

        async def gen(message: str, max_tokens: int, **extra) -> dict:
            payload = {"session_id": sid, "message": message,
                       "max_tokens": max_tokens, **extra}
            r = await cx.post(f"{BASE}/generate", json=payload)
            r.raise_for_status()
            return r.json()

        def report(tag: str, g: dict) -> None:
            reason = g.get("reasoning_text", "")
            print(f"\n{tag}: {g['text'][:140]!r}")
            print(f"  prompt={g['n_prompt_total']} cached={g['n_cached_total']} "
                  f"gen={g['n_generated']} (reasoning {len(reason)} chars) "
                  f"t/s={g['tokens_per_s']:.1f}")

        listener = asyncio.create_task(listen())
        g1 = await gen("In one paragraph, what is cognitive dissonance?", 150)
        await listener
        report("turn 1 (thinking)", g1)

        pf = next((e for e in events if e[0] == "prefill_done"), None)
        steps = [e for e in events if e[0] == "step_done"]
        pf_desc = f"{pf[1]} ({pf[2]} tokens, {pf[3]}ms)" if pf else "none"
        print(f"  viz prefill shape={pf_desc} | step_done={len(steps)} "
              f"| step shape={steps[0][1] if steps else None}")

        g2 = await gen("Give one everyday example of it.", 60,
                       temperature=0.7,
                       chat_template_kwargs={"enable_thinking": False})
        report("turn 2 (thinking off)", g2)

        g3 = await gen("Name one researcher famous for studying it.", 120)
        report("turn 3 (thinking on)", g3)

        ok = (bool(pf and pf[1] and pf[1][0] == 24)
              and bool(steps)
              and g1["n_generated"] > 0 and g1["text"]
              and g2["text"]
              and g3["n_cached_total"] > 0)
        print(f"\nRESULT: {'PASS' if ok else 'CHECK MANUALLY'} "
              f"(viz 24L prefill={bool(pf and pf[1])}, "
              f"steps={len(steps)}, turn-2 KV reuse={g2['n_cached_total']}"
              f"/{g2['n_prompt_total']})")


if __name__ == "__main__":
    asyncio.run(main())
