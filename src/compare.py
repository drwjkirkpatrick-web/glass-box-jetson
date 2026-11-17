#!/usr/bin/env python3
"""
Phase 4: FP16 vs INT4 activation comparison.

Loads the FP16 1.5B model (heavier — only runs in this comparison mode),
hooks every decoder block's residual stream, runs the same prompt through
both:
  - FP16 model via transformers/torch (hooked)
  - INT4 model via llama-cpp-python (or llama-server HTTP if loaded)

Computes layer-by-layer cosine similarity between FP16 and INT4 hidden
states, and emits a report showing where the quantization distortion
concentrates.

This is meant to be run manually, NOT during normal serving, because
loading the FP16 1.5B model needs ~3 GB on top of the running INT4 server.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent


def _get_decoder_blocks(model) -> torch.nn.ModuleList:
    for attr in ("model", "transformer"):
        if hasattr(model, attr):
            inner = getattr(model, attr)
            for inner_attr in ("layers", "h", "blocks"):
                if hasattr(inner, inner_attr):
                    return getattr(inner, inner_attr)
    raise RuntimeError(f"Could not locate decoder blocks on {type(model).__name__}")


@torch.no_grad()
def fp16_layer_activations(model, tokenizer, prompt: str, device: str) -> list[np.ndarray]:
    """Run prompt through FP16 model, return list of [seq, hidden] arrays, one per layer."""
    layer_acts: dict[int, list[torch.Tensor]] = {}

    def make_hook(li):
        def h(m, i, o):
            t = o[0] if isinstance(o, tuple) else o
            layer_acts.setdefault(li, []).append(t[0].detach().cpu().float())
        return h

    blocks = _get_decoder_blocks(model)
    handles = [blocks[i].register_forward_hook(make_hook(i)) for i in range(len(blocks))]
    try:
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        out = model.generate(ids, max_new_tokens=1, do_sample=False)
    finally:
        for h in handles:
            h.remove()

    # Take only the prompt-length portion (not the generated token)
    seq_len = ids.shape[1]
    out = []
    for li in range(len(blocks)):
        # Concatenate and slice to prompt length
        acts = torch.cat(layer_acts[li], dim=0)[:seq_len]
        out.append(acts.numpy())
    return out


def int4_layer_activations_via_cli(gguf_path: Path, prompt: str, layers: int) -> list[np.ndarray]:
    """
    Fall back to using llama.cpp's `llama-embedding` or `llama-perplexity` to
    extract intermediate states from the INT4 GGUF. This is hacky and
    probably not worth the effort in practice — INT4 internal states
    aren't easily extractable from llama.cpp.

    So instead, for the INT4 side of the comparison we just generate text
    and compare to the FP16-generated text token-by-token. The
    layer-by-layer diff is only computed on the FP16 side; the INT4
    side is characterized by generation quality (perplexity, exact match)
    on the same prompt.
    """
    # Run llama-cli to generate, capture output
    cli = REPO_ROOT / "llama.cpp" / "build" / "bin" / "llama-cli"
    if not cli.exists():
        return []
    try:
        result = subprocess.run(
            [str(cli), "-m", str(gguf_path), "-p", prompt, "-n", "64",
             "--no-display-prompt", "-ngl", "99", "-c", "2048"],
            capture_output=True, text=True, timeout=120,
        )
        out = result.stdout.strip()
    except Exception as e:
        print(f"[int4] llama-cli failed: {e}", file=sys.stderr)
        return []
    # We return an empty list here — the actual token text is in `out`.
    # Caller compares text-level metrics.
    return [{"text": out}] if out else []


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    ab = (a * b).sum(axis=-1)
    return float((ab / (na * nb + 1e-9)).mean())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fp16-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--int4-model", required=True, help="Path to GGUF Q4_K_M file")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--out", default=str(REPO_ROOT / "reports" / "fp16_vs_int4.json"))
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading FP16 {args.fp16_model} on {device}...")
    tok = AutoTokenizer.from_pretrained(args.fp16_model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.fp16_model,
        torch_dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval()
    print(f"  GPU mem: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    print(f"Running FP16 forward on prompt: {args.prompt!r}")
    t0 = time.time()
    fp16_acts = fp16_layer_activations(model, tok, args.prompt, device)
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  {len(fp16_acts)} layers, activation shape per layer: {fp16_acts[0].shape}")

    # Per-layer norm & basic stats (FP16)
    norms = [float(np.linalg.norm(a, axis=-1).mean()) for a in fp16_acts]
    print("\nFP16 per-layer activation L2 norm (mean over tokens):")
    for i, n in enumerate(norms):
        print(f"  layer {i:2d}: {n:.2f}")

    # Run INT4 model and capture text
    print(f"\nRunning INT4 {args.int4_model} on same prompt...")
    int4 = int4_layer_activations_via_cli(Path(args.int4_model), args.prompt, len(fp16_acts))
    int4_text = int4[0]["text"] if int4 else ""

    # FP16 generate for text-level comparison
    print("Running FP16 generation for text comparison...")
    inputs = tok(args.prompt, return_tensors="pt").to(device)
    fp16_out = model.generate(**inputs, max_new_tokens=64, do_sample=False)
    fp16_text = tok.decode(fp16_out[0], skip_special_tokens=True)

    # Save report
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "prompt": args.prompt,
        "fp16_activations": {
            "num_layers": len(fp16_acts),
            "seq_len": int(fp16_acts[0].shape[0]),
            "hidden_dim": int(fp16_acts[0].shape[1]),
            "per_layer_norm": norms,
            # We don't save the full activation matrices — just summary stats.
            "per_layer_mean": [a.mean(axis=0).tolist() for a in fp16_acts[:1]],  # first layer only as sample
        },
        "fp16_text": fp16_text,
        "int4_text": int4_text,
        "notes": [
            "INT4 internal activations are not extractable from llama.cpp.",
            "Comparison is at the OUTPUT text level only.",
            "For layer-level INT4 distortion, use llama.cpp's --verbose flag or",
            "extract hidden states via the C++ API directly.",
        ],
    }
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {args.out}")
    print(f"\n=== FP16 output ===\n{fp16_text}\n")
    if int4_text:
        print(f"=== INT4 output ===\n{int4_text}\n")


if __name__ == "__main__":
    main()
