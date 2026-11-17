# Glass Box Jetson

Watching a local LLM think on the Jetson Orin Nano 8GB — any model in the zoo,
served INT4 under llama.cpp, with its FP16 twin hooked for live per-layer
activation visualization, session-pinned KV-cache telemetry, and
reasoning-token streaming for thinking models.

Adapted from [mohith-das/glass_box](https://github.com/mohith-das/glass_box)
(MIT), which ran a fixed Qwen2.5-1.5B/0.5B pair. This fork generalizes it to
other models, including hybrid-attention VLM twins, and fixes the parts that
broke on a real agentic model.

## What the dashboard shows (:8001)

1. **Tokens streaming** — reasoning tokens arrive dimmed/italic for thinking
   models; the answer text renders after the thinking phase.
2. **A 24×32 layer grid lighting up** — per-layer PCA-projected activations of
   the FP16 twin on the same text, per token, live (the grid size adapts to
   the twin's layer count).
3. **Cached vs fresh** — which prompt tokens were served from the
   session-pinned KV cache, per request and cumulative.

## Verified run (2026-11-16)

Served + visualized **BotPsy-0.8B** (qwen3_5 hybrid GatedDeltaNet, thinking):

- served: `botpsy-0.8b-q4_k_m-imat.gguf` — llama-server (CUDA, sm_87, `-ngl 99`, ctx 2048)
- viz twin: `qvac/BotPsy-0.8B` FP16 on CUDA — `AutoModelForImageTextToText`
  with forward hooks on the language-model decoder blocks
- PCA basis: 24 layers × 32 components, explained variance 0.51–0.93
  (one-time calibration, 20 prompts)

| Turn | Mode | Generated | Reasoning | KV reuse |
|---|---|---|---|---|
| 1 | thinking on | 146 tok | 643 chars + answer | 23/44 |
| 2 | thinking off (`chat_template_kwargs`) | 30 tok, instant answer | 0 chars | 40/82 |
| 3 | thinking on | 120 tok | 556 chars | 78/128 |

- viz events: prefill `[24, 30, 32]` (1953 ms), 146 `step_done` events —
  the heatmap lights up during the thinking phase
- effective 4.0–4.8 tok/s including per-token viz round-trips
  (llama.cpp alone benches ~54 tok/s on this Q4)
- stack healthy in ~21 s; ~7.1 GB of 7.4 GB in use with the OS baseline —
  view the dashboard from another device on the LAN, not from a browser on
  the Jetson itself

## Adaptations vs upstream

| Change | Why |
|---|---|
| `--llama-bin` | point at an existing CUDA llama-server build — no in-repo clone/build |
| `--viz-python` | run the viz model in a CUDA-torch venv while the controller stays on system python |
| `--llama-extra-args` | carry champion launch flags (`--cache-type-k`, `-fa`, …) |
| per-request sampling (`temperature`/`top_k`/`top_p`/`repeat_penalty`) + CLI defaults | vendor sampling per model, per call |
| `--chat-template-kwargs` + per-request override | toggle thinking per model or per turn |
| `reasoning_content` streaming | thinking models stream reasoning first; it is surfaced, flagged, fed to the heatmap, and kept out of history |
| VLM-aware viz loader + block walker | Qwen3.5 `ForConditionalGeneration` twins (decoder at `model.model.language_model.layers` or the top-level shim) |
| contextual steps via `past_key_values` | upstream fed each streamed token through a context-free forward; for hybrid GatedDeltaNet layers that misrepresents the activations |
| dynamic `/v1/models` + dashboard title | no hardcoded Qwen2.5 strings |

## Adding a model from the zoo

1. **Serve**: pass the GGUF via `--model` (any llama.cpp-supported arch).
2. **Viz twin**: an HF FP16 twin via `--viz-model` (must be
   transformers-loadable; VLM wrappers are auto-detected).
3. **Calibrate once**:
   `python src/neuron_clusters.py --model <hf-id> --out models/neuron_clusters_<name>.json`
4. **Vendor sampling**: `--temperature/--top-k/--top-p/--repeat-penalty` at
   launch, overridable per request; thinking toggle via
   `--chat-template-kwargs '{"enable_thinking": false}'`.
5. **Extra llama-server flags**: `--llama-extra-args`.

Memory rule for 8 GB: the viz twin must stay ≤ ~1.5B FP16 next to a ≤3B Q4
served model. For 8B+ served models, use a small same-family viz proxy
(upstream's approach) or run the viz solo.

## Launch

```bash
scripts/launch_botpsy.sh          # the exact verified configuration
# or directly:
/usr/bin/python3.10 src/server.py \
  --model ~/models/new-zoo/botpsy-0.8b-q4_k_m-imat.gguf \
  --llama-bin ~/llama.cpp/build/bin/llama-server \
  --llama-port 8080 --controller-port 8001 -ngl 99 --ctx 2048 \
  --viz-model qvac/BotPsy-0.8B \
  --viz-python ~/.venvs/jetson-pytorch/bin/python \
  --viz-clusters models/neuron_clusters_botpsy.json \
  --temperature 0.7
```

Dashboard: `http://<jetson-ip>:8001/` · smoke test: `scripts/e2e_test.py`.

## Environment (verified)

- Jetson Orin Nano Super 8GB — JetPack 6 (L4T R36.x), CUDA 12.6, sm_87
- llama-server 1.0.0-dev (build 1995), `GGML_CUDA`
- controller: system python 3.10 — fastapi 0.143, uvicorn, websockets 16.1,
  httpx, psutil
- viz venv: torch 2.5.0a0+nv24.8 (CUDA), transformers 5.14.1 (qwen3_5 needs
  ≥5.14), tokenizers 0.22.2, huggingface_hub 1.24.0
- flash-linear-attention not installed → hybrid layers run the torch
  fallback (works, slower)

## Caveats

- ~230 MB RAM headroom with this stack — the viz watchdog respawns an
  OOM-killed twin in 2–3 s, and the heatmap goes dark meanwhile.
- The viz twin shares the served model's weights modulo quantization, but
  activations come from the twin's own forward pass — not the INT4 engine's
  internal states (that needs C++ instrumentation inside llama.cpp).
- NVML queries are NotSupported on the Orin iGPU, so `gpu_mb` reads 0; use
  `tegrastats`/`jetson-stats` for GPU memory.

## Attribution

Adapted from [mohith-das/glass_box](https://github.com/mohith-das/glass_box)
(MIT; original README kept at `docs/ORIGINAL_README.md`, license preserved).
Inference by [llama.cpp](https://github.com/ggml-org/llama.cpp).
BotPsy-0.8B by qvac (Apache 2.0, Qwen3.5-0.8B base).
