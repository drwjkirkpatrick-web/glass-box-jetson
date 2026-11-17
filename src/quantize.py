#!/usr/bin/env python3
"""
Phase 1: Quantize Qwen2.5-1.5B-Instruct from HuggingFace FP16 to GGUF Q4_K_M.

This uses llama.cpp's built-in conversion tool (convert_hf_to_gguf.py) plus
the C++ llama-quantize binary. The output is a single .gguf file that
llama.cpp (and llama-server) can load with GPU offload.

Memory budget on Jetson Orin Nano Super (8GB shared CPU/GPU):
    FP16 weights: ~3.0 GB  -> cannot fit + activations + cache + OS
    Q4_K_M GGUF:  ~1.1 GB  -> fits with ~1.5GB headroom for KV cache + viz buffers
"""
import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LLAMA_CPP_DIR = REPO_ROOT / "llama.cpp"
HF_CACHE = REPO_ROOT / "models" / "hf"
GGUF_DIR = REPO_ROOT / "models" / "gguf"


def find_snapshot(model_id: str) -> Path:
    """Find the local snapshot dir for a HuggingFace model id (e.g. 'Qwen/Qwen2.5-1.5B-Instruct')."""
    safe_id = model_id.replace("/", "--")
    snap_root = HF_CACHE / f"models--{safe_id}" / "snapshots"
    if not snap_root.exists():
        raise FileNotFoundError(
            f"No snapshot for {model_id} at {snap_root}. Run download first."
        )
    snapshots = sorted(snap_root.iterdir(), key=lambda p: p.stat().st_mtime)
    if not snapshots:
        raise FileNotFoundError(f"No snapshots under {snap_root}")
    snap = snapshots[-1]
    if not (snap / "model.safetensors").exists():
        raise FileNotFoundError(f"Snapshot {snap} missing model.safetensors")
    return snap


def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(cmd)}  (cwd={cwd or os.getcwd()})")
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0:
        sys.exit(f"command failed with exit code {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="Qwen/Qwen2.5-1.5B-Instruct",
        help="HF model id, must already be in models/hf",
    )
    parser.add_argument(
        "--quant", default="Q4_K_M",
        help="GGUF quantization type (Q4_K_M, Q4_0, Q5_K_M, Q8_0, F16, ...)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output GGUF path (default: models/gguf/<model>-<quant>.gguf)",
    )
    args = parser.parse_args()

    snap = find_snapshot(args.model)
    model_short = args.model.split("/")[-1]
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: HF safetensors -> GGUF FP16 (intermediate)
    fp16_gguf = GGUF_DIR / f"{model_short}-f16.gguf"
    if not fp16_gguf.exists():
        convert_script = LLAMA_CPP_DIR / "convert_hf_to_gguf.py"
        if not convert_script.exists():
            sys.exit(f"missing {convert_script}; did the llama.cpp clone complete?")
        run([
            sys.executable, str(convert_script),
            str(snap),
            "--outfile", str(fp16_gguf),
            "--outtype", "f16",
        ])
    else:
        print(f"[skip] {fp16_gguf} exists")

    # Step 2: FP16 GGUF -> Q4_K_M GGUF (or whatever quant was requested)
    out_gguf = Path(args.out) if args.out else GGUF_DIR / f"{model_short}-{args.quant}.gguf"
    if out_gguf.exists():
        print(f"[skip] {out_gguf} exists")
    else:
        quantize_bin = LLAMA_CPP_DIR / "build" / "bin" / "llama-quantize"
        if not quantize_bin.exists():
            sys.exit(
                f"missing {quantize_bin}. Build llama.cpp first:\n"
                f"  cd llama.cpp && cmake -B build -DGGML_CUDA=ON && cmake --build build --target llama-server"
            )
        run([str(quantize_bin), str(fp16_gguf), str(out_gguf), args.quant])

    print(f"\nDone. Quantized model at: {out_gguf}")
    print(f"  size: {out_gguf.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
