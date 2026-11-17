#!/usr/bin/env python3
"""
Compute neuron-cluster projections for the activation visualization.

We run the FP16 viz model (Qwen2.5-0.5B-Instruct) over a small calibration
corpus, hook every decoder layer's residual stream, and do PCA per layer to
find the directions that explain the most variance. The first K components
become the "neuron clusters" the dashboard plots.

This is computed ONCE and saved to models/neuron_clusters.json. At inference
time the viz model just projects new activations through this matrix.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VIZ_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUT = REPO_ROOT / "models" / "neuron_clusters.json"

# A small calibration corpus. These are generic English/natural-language prompts
# designed to light up a wide variety of features. They are NOT part of the
# actual demo; they only seed the PCA basis.
CALIBRATION_PROMPTS = [
    "The capital of France is",
    "Once upon a time in a land far away,",
    "In the year 2050, technology will",
    "The function f(x) = x^2 has derivative",
    "Photosynthesis is the process by which plants",
    "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(",
    "SELECT * FROM users WHERE active =",
    "Q: What is the speed of light?\nA:",
    "To bake a chocolate cake, first preheat the oven to",
    "The three branches of the US government are",
    "DNA stands for deoxyribonucleic acid, which",
    "import numpy as np\narr = np.array([1, 2, 3])\nprint(arr.",
    "The Roman Empire fell in",
    "Climate change is caused primarily by",
    "In machine learning, overfitting occurs when",
    "The mitochondria is the powerhouse of",
    "To solve this equation, we first isolate",
    "Shakespeare wrote many plays including Hamlet,",
    "The chemical symbol for gold is",
    "When debugging code, the first step is to",
]

NUM_CLUSTERS = 32  # columns in the dashboard grid per layer


@torch.no_grad()
def collect_activations(model, tokenizer, prompts: list[str], device: str) -> list[np.ndarray]:
    """Run each prompt through the model, collecting per-layer residual stream activations."""
    layer_acts: dict[int, list[np.ndarray]] = {}

    def make_hook(layer_idx: int):
        def hook(module, inputs, outputs):
            # The output of a decoder block is a tuple; first element is hidden states.
            hidden = outputs[0] if isinstance(outputs, tuple) else outputs
            # Take the last-token hidden state, mean over batch, store.
            # Shape: [batch, seq, hidden] -> mean over batch+seq -> [hidden]
            flat = hidden.detach().float().reshape(-1, hidden.shape[-1])
            layer_acts.setdefault(layer_idx, []).append(flat.cpu().numpy())
        return hook

    # Register hooks on every decoder block
    blocks = _get_decoder_blocks(model)
    handles = [blocks[i].register_forward_hook(make_hook(i)) for i in range(len(blocks))]

    try:
        for prompt in prompts:
            inputs = tokenizer(prompt, return_tensors="pt").to(device)
            model(**inputs)
    finally:
        for h in handles:
            h.remove()

    # Stack per layer: [num_prompts*samples, hidden]
    n_layers = len(blocks)
    out = []
    for li in range(n_layers):
        acts = np.concatenate(layer_acts[li], axis=0)
        out.append(acts)
    return out


def _get_decoder_blocks(model) -> torch.nn.ModuleList:
    """Locate the decoder-block ModuleList regardless of model architecture:
    dense LMs (Qwen2/Llama/Mistral, GPT-2) and multimodal wrappers whose text
    decoder sits at model.model.language_model.layers or behind a top-level
    model.language_model shim."""
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


def pca_basis(acts: np.ndarray, k: int) -> np.ndarray:
    """Return [hidden, k] projection matrix: top-k principal directions."""
    acts = acts - acts.mean(axis=0, keepdims=True)
    # SVD: acts = U S Vt. Rows of Vt are principal directions.
    # acts is [N, hidden]; we want components over the hidden dim.
    _, _, Vt = np.linalg.svd(acts, full_matrices=False)
    k = min(k, Vt.shape[0])
    return Vt[:k].T  # [hidden, k]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_VIZ_MODEL)
    parser.add_argument("--out", default=str(DEFAULT_OUT))
    parser.add_argument("--clusters", type=int, default=NUM_CLUSTERS)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.model} on {device} for PCA calibration...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    model_cls = (
        AutoModelForImageTextToText
        if getattr(cfg, "vision_config", None) is not None
        else AutoModelForCausalLM
    )
    model = model_cls.from_pretrained(
        args.model,
        dtype=torch.float16,
        trust_remote_code=True,
    ).to(device).eval()

    print(f"Collecting activations over {len(CALIBRATION_PROMPTS)} calibration prompts...")
    per_layer_acts = collect_activations(model, tokenizer, CALIBRATION_PROMPTS, device)
    n_layers = len(per_layer_acts)
    hidden_dim = per_layer_acts[0].shape[1]
    print(f"  -> {n_layers} layers, hidden_dim={hidden_dim}")

    print(f"Computing PCA per layer with k={args.clusters}...")
    bases = np.zeros((n_layers, hidden_dim, args.clusters), dtype=np.float32)
    explained = []
    for li, acts in enumerate(per_layer_acts):
        bases[li] = pca_basis(acts, args.clusters)
        # Reconstruction variance explained
        centered = acts - acts.mean(axis=0, keepdims=True)
        proj = centered @ bases[li]
        recon = proj @ bases[li].T
        var = float(np.linalg.norm(recon) ** 2) / float(np.linalg.norm(centered) ** 2 + 1e-9)
        explained.append(var)
        print(f"  layer {li:2d}: variance explained by {args.clusters} PCs = {var:.3f}")

    # Save projection bases + metadata
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": args.model,
        "num_layers": n_layers,
        "hidden_dim": hidden_dim,
        "num_clusters": args.clusters,
        "explained_variance": explained,
        # Save as float16 to halve the file size on disk (still plenty of precision for viz).
        "bases_f16": bases.astype(np.float16).tobytes(),
    }
    # ~hidden_dim * num_clusters * 2 bytes/layer * ~24 layers = ~1MB
    print(f"\nSaving to {out_path} ({os.path.getsize(out_path) if out_path.exists() else 0} -> ~{bases.nbytes/1e6:.1f} MB)")
    with open(out_path, "wb") as f:
        head = json.dumps({k: v for k, v in payload.items() if k != "bases_f16"}).encode()
        f.write(head)
        f.write(b"\n")
        f.write(payload["bases_f16"])
    print("Done.")


if __name__ == "__main__":
    main()
