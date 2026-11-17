# Glass Box

> A transparent local LLM inference server on an NVIDIA Jetson Orin Nano Super (8 GB).
> Watches INT4 quantization, session-pinned KV cache reuse, and per-layer
> activation patterns all in one place — and shows you, live, which tokens
> came from the cache and which were freshly computed.

![Jetson Orin Nano Super](https://img.shields.io/badge/Jetson-Orin%20Nano%20Super-76B900?style=flat-square&logo=nvidia)
![CUDA 12.6](https://img.shields.io/badge/CUDA-12.6-76B900?style=flat-square&logo=nvidia)
![Python 3.10](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python)
![llama.cpp](https://img.shields.io/badge/llama.cpp-HEAD-blue?style=flat-square)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## What is this?

Most LLM demos show one thing at a time: quantization, or KV caching, or
interpretability. Glass Box shows all three bound together, visibly, on
real hardware with a hard memory budget that forces real tradeoffs.

You type a prompt into a web dashboard. The dashboard shows:

1. **Tokens streaming in** — the output of a Qwen2.5-1.5B model running
   in **INT4 (GGUF Q4_K_M)** under llama.cpp.
2. **A 24×32 layer grid lighting up** — per-layer, per-PCA-cluster
   activation magnitudes on the *same* text from a second, **FP16
   0.5B** "shadow" model with forward hooks attached. Color is the
   delta from the previous token.
3. **A live "cached vs fresh" indicator** — which prompt tokens were
   served from the session's pinned KV cache, and which required a
   fresh forward pass. This comes from llama.cpp's
   `usage.prompt_tokens_details.cached_tokens` field, surfaced per
   request and cumulative across the session.

The three are connected: the cached portion of the prompt is the same
text the visualization model is also processing, so you can see what
*its* activations look like on the prefix the INT4 model didn't have
to re-compute.

The 8 GB memory ceiling isn't a footnote — it's the design constraint
that forces the dual-model split (more on this below).

## Hardware

Tested on:

- **NVIDIA Jetson Orin Nano Super** (JetPack 6.x, CUDA 12.6)
- 8 GB unified CPU/GPU memory (CPU and GPU share one pool)
- 6× Cortex-A78AE CPU cores (ARMv8.2)
- 1024 CUDA cores, Ampere SM 8.7
- 233 GB NVMe

At idle the OS uses ~3.2 GB. The 4.5 GB that remains has to hold:

| Component                     | RAM        |
|-------------------------------|------------|
| llama-server + INT4 1.5B (CPU)| ~1.0 GB    |
| FastAPI controller            | ~0.1 GB    |
| Viz model + FP16 0.5B         | ~2.0–2.5 GB (CUDA + host pinned) |
| Browser/buffers               | ~0.3 GB    |
| **Headroom**                  | ~0.5–1.0 GB |

It's tight. The viz model has a watchdog that respawns it if OOM
kills it under load (see "Caveats" below).

## Quick start

```bash
git clone https://github.com/mohith-das/glass_box.git
cd glass_box
./start.sh
```

Then open **http://localhost:8001/** in a browser.

`start.sh` handles everything:

1. Clones and builds llama.cpp from source (CPU mode by default, ~5 min).
2. Downloads and quantizes Qwen2.5-1.5B-Instruct to GGUF Q4_K_M
   (3 GB FP16 → 1 GB INT4).
3. Downloads Qwen2.5-0.5B-Instruct and computes the PCA basis for the
   visualization (one-time, ~2 min).
4. Starts three services and connects them together.
5. Logs go to `logs/`.

Tunables (set as env vars before `./start.sh`):

| Var            | Default              | Effect                                          |
|----------------|----------------------|-------------------------------------------------|
| `WITH_CUDA`    | `OFF`                | Build llama.cpp with CUDA (~1-2 hour build)     |
| `N_GPU_LAYERS` | `99`                 | Layers of the INT4 model to offload to GPU      |
| `CTX`          | `2048`               | Context window per slot                         |
| `MAX_SESSIONS` | `4`                  | Concurrent KV-cached sessions                   |
| `CONTROLLER_PORT` | `8001`            | Override controller port                        |

For the full GPU build (recommended for real inference speed):

```bash
./rebuild_cuda.sh          # ~1-2 hour background build
WITH_CUDA=ON ./start.sh    # once the build finishes
```

## Architecture

```
                 ┌─────────────────────────┐
                 │       dashboard         │
                 │       (browser)         │
                 │                         │
                 │  POST /session          │
                 │  POST /generate         │
                 │  /ws/dashboard  ◄───────┼── tokens, cache stats,
                 │  /ws/viz  (deprecated)  │   memory, viz events
                 │  GET  /                 │
                 │  GET  /config           │
                 └──────────┬──────────────┘
                            │ HTTP / WebSocket
                ┌───────────┼────────────┐
                │           │            │
                ▼           ▼            ▼
       ┌─────────────┐  ┌──────────┐  ┌────────────────┐
       │ Controller  │  │ llama-   │  │  Viz model     │
       │  (FastAPI)  │──│  server  │  │  (FP16 0.5B,   │
       │  :8001      │  │  :8080   │  │   forward      │
       │             │  │          │  │   hooks, CUDA) │
       │ session mgmt│  │ INT4 1.5B│  │  :8765         │
       │ OAI proxy   │  │ slot KV  │  │                │
       │ cache stats │  │ cache    │  │ WebSocket out  │
       │ viz client  │  │          │  │                │
       └──────┬──────┘  └──────────┘  └────────┬───────┘
              │                                  │
              │ internal WebSocket:             │ WebSocket:
              │ prefill, step,                   │ prefill, step,
              │ broadcast results                │ step_done
              │ to /ws/dashboard                 │
              ▼                                  ▼
       /ws/dashboard clients  ◄────  /ws/viz (viz_model direct)
```

The controller is the orchestrator. It owns the llama-server
supervisor and the viz model supervisor, talks to both, and merges
their outputs into a single WebSocket stream that the dashboard
listens to.

## How each phase works

### Phase 1 — Quantization (`src/quantize.py`)

Converts Qwen2.5-1.5B-Instruct from HuggingFace safetensors to
GGUF, then quantizes to Q4_K_M using llama.cpp's `llama-quantize`.
The intermediate FP16 GGUF is 3 GB on disk; the Q4_K_M output is
0.99 GB. The 0.75 GB savings over FP16 is what makes the rest of
the system fit in 8 GB.

### Phase 2 — KV-cached session serving (`src/server.py`)

The controller pins one llama-server **slot** per session. Every
`/generate` call sends the **full conversation history** as the
OAI `messages` array. When the prefix matches what the slot already
has in KV cache, llama-server reuses those cached keys and values
and only freshly computes forward passes for the new tokens.

The OpenAI `usage.prompt_tokens_details.cached_tokens` field tells
us exactly how many prompt tokens were served from cache. We surface
that per request and cumulative across the session. Real numbers
from a three-turn session on Jetson:

```
Turn 1: cached 20/32 (62%) —  3.3 tok/s
Turn 2: cached 39/56 (70%) —  3.4 tok/s
Turn 3: cached 63/80 (79%) —  4.5 tok/s
```

Cache hit rate climbs each turn as the prefix accumulates, and
generation speed climbs with it (cached prompt tokens are
substantially cheaper than freshly-computed ones).

**Critical detail**: the full conversation history must be re-sent
on every call. If you send only the new user message, llama-server
sees a prompt that doesn't match the slot's existing state, cache
hits drop to zero, and on small quantized models this manifests as
**runaway token loops** (the model generates from a mismatched
position with no end-of-sequence in sight). We learned this the
hard way.

### Phase 3 — Live activation visualization (`src/viz_model.py` + `src/neuron_clusters.py`)

A separate Python process runs Qwen2.5-0.5B-Instruct in FP16 on
CUDA, with PyTorch **forward hooks** attached to every decoder
block's residual stream. For each prompt, it:

1. Captures `[seq, hidden]` activations per layer.
2. Projects each layer's activations through a precomputed PCA
   basis (32 directions per layer, fit offline on a 20-prompt
   calibration set in `src/neuron_clusters.py`).
3. Streams `[24 layers, 32 clusters]` activations over a WebSocket
   to the controller, which re-broadcasts them on `/ws/dashboard`.

The controller owns a **single persistent WebSocket** to the viz
model. On every `/generate` call, the controller:

- Sends `{"kind": "prefill", "prompt": ...}` to the viz model
  when a new user turn starts, awaits the response, and broadcasts
  the activation data.
- Sends `{"kind": "step", "token_id": ...}` for each generated
  token, and broadcasts the result.

The viz model has a watchdog (`VizModelSupervisor._watchdog_loop`
in `src/server.py`) that respawns the process within 2-3 seconds
if it dies — a real concern at this memory budget.

The viz model has a **different tokenizer** than the served
model (0.5B vs 1.5B, though both Qwen2.5 family). The controller
asks the viz model to tokenize the growing text and feeds back
its own token ids. The visualization is a "what the 0.5B model
sees on the same text" proxy, not a bit-exact replay of the 1.5B
compute. This is the cleanest separation that fits in 8 GB.

### Phase 4 — FP16 vs INT4 comparison (`src/compare.py`)

A standalone script that loads Qwen2.5-1.5B-Instruct in FP16
(heavier — ~3 GB) and runs the same prompt through both
quantizations, capturing per-layer FP16 activation norms. Note:
extracting **internal hidden states** from llama.cpp's INT4 path
requires C++ instrumentation, which we don't do here. The
comparison is at the **output text + activation norm** level, not
at the hidden-state diff level.

## The honest tradeoff: why two models

A single 1.5B FP16 model is ~3 GB just for weights. Add activations
for a 2048-token sequence, KV cache for batched inference, browser,
OS, Python — you're past 8 GB before you start rendering.

Quantization to INT4 cuts the weights to ~1 GB, which is what
makes the INT4 server tractable. But quantization also cuts the
model's effective precision; small differences in weights get
amplified through 24 transformer layers.

A *second* FP16 model for visualization doubles the memory cost
and defeats the point of quantization. We solve this by using a
**smaller model (0.5B)** for the viz. The visualization is honest
about being a different model on the same text, not a transparent
view into the INT4 model itself.

To make the visualization *actually* transparent into the INT4
model, you'd need a custom inference path that exposes hooks —
either by running llama.cpp's compute graph through a Python-aware
backend, or by maintaining a parallel PyTorch INT4 model. The
first doesn't really exist; the second doubles the memory again
and breaks the demo on 8 GB. **That's the tradeoff this project
exposes.**

## Repository layout

```
glass_box/
├── README.md                  this file
├── start.sh                   one-command startup
├── rebuild_cuda.sh            helper for the ~1-2 hour CUDA build
├── requirements.txt
├── .gitignore
├── LICENSE                    MIT
├── frontend/
│   └── index.html             dashboard (vanilla JS, no build step)
├── src/
│   ├── quantize.py            HF → GGUF FP16 → GGUF Q4_K_M
│   ├── neuron_clusters.py     PCA calibration for the viz model
│   ├── viz_model.py           FP16 0.5B + forward hooks, WebSocket
│   ├── server.py              controller: sessions, OAI proxy, viz supervisor
│   └── compare.py             FP16 vs INT4 diff (Phase 4, manual)
├── logs/                      runtime logs (gitignored)
└── llama.cpp/                 cloned and built from source (gitignored)
```

After `./start.sh` runs once, you'll also have:

- `models/hf/...`              HuggingFace cache for the FP16 safetensors
- `models/gguf/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf`  quantized model (~1 GB)
- `models/neuron_clusters.json`                     precomputed PCA basis

## Caveats

- **Memory is tight.** All three services + the browser use
  ~7.5 GB. If you add anything else (jtop, IDE, additional
  browser tabs), you risk OOM. The viz model is the most
  likely victim: a long prefill or a large session can push it
  over the edge, and Linux's OOM killer terminates it without
  warning. The watchdog in `VizModelSupervisor` recovers
  automatically, but the heatmap will go dark for ~2-3 seconds
  during a restart.
- **The viz model is 0.5B; the served model is 1.5B.** They will
  sometimes disagree on what tokens "matter". The visualization
  is a useful proxy, not a ground-truth view of the 1.5B's
  internals.
- **The INT4 model runs on CPU by default.** ~3-5 tok/s is
  demoable but slow. `WITH_CUDA=ON ./start.sh` enables GPU
  offload (~20-40 tok/s on Orin), but the CUDA build takes
  1-2 hours on Jetson ARM and there are no prebuilt wheels
  available (see "CUDA build, briefly explained" below).
- **The viz grid is "delta from previous token", not raw
  magnitude.** We scale per layer by max-abs to keep colors
  meaningful, but the colorbar isn't an objective activation
  measure — it's a relative signal.
- **CORS:** the controller binds on `0.0.0.0` so the dashboard
  works from any host on the network. llama-server and the
  viz model bind on `127.0.0.1` for safety. The frontend's
  `/config` endpoint returns the correct `ws://` URL for
  whatever host the dashboard is loaded from.

## CUDA build, briefly explained

If you want the INT4 model running on the Orin's CUDA cores,
you need a CUDA-enabled `llama-server`. There are no
pre-built Jetson wheels for `llama-cpp-python` or llama.cpp
binaries on PyPI, HuggingFace, GitHub releases, or NVIDIA NGC.
The community standard is to build from source.

On Jetson ARM with CUDA, this build takes 1-2 hours because:

- ~70 CUDA files are compiled by `nvcc`, which is single-threaded
  per file.
- `nvcc` on Jetson ARM is ~5× slower than on x86 (the cross-
  compilation path through `nvcc`'s internal device compiler
  is CPU-bound and not optimized for ARM).
- llama.cpp has ~90 explicit flash-attention / matrix-multiply
  template instantiations (one `.cu` file per `(head_dim,
  block_size, dtype)` combo). Each pays the full `nvcc` cost.
- The flash-attention "MMA" templates target Hopper SM 9.0+;
  they aren't used on Orin SM 8.7, but they're still required
  at link time, so they all have to compile.

`./rebuild_cuda.sh` runs the build in the background and logs
to `/tmp/glassbox_cuda_build.log`. `tail -f` that file to watch
progress.

## Development

To work on the code without re-downloading models on every
restart, the first `./start.sh` will leave the HF safetensors,
the quantized GGUF, and the neuron clusters on disk. Subsequent
runs reuse them.

To reset state:

```bash
rm -rf models/      # re-download + re-quantize
rm -rf llama.cpp/   # re-clone + re-build
```

## License

MIT. See [LICENSE](LICENSE).

## Acknowledgments

- [llama.cpp](https://github.com/ggerganov/llama.cpp) — the
  inference backend.
- [Qwen2.5](https://huggingface.co/Qwen) — the models.
- [FastAPI](https://fastapi.tiangolo.com/) and
  [uvicorn](https://www.uvicorn.org/) — the controller.
- The [Jetson community](https://www.jetson-ai-lab.com/) —
  build recipes and patience.
