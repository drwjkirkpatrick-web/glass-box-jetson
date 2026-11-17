#!/usr/bin/env bash
# One-command startup for Glass Box.
#
# Steps:
#   1. Build llama.cpp (CPU or CUDA) if llama-server binary missing
#   2. Quantize Qwen2.5-1.5B-Instruct to Q4_K_M GGUF if not present
#   3. Compute neuron PCA clusters (one-time calibration) if not present
#   4. Start the viz model (FP16 0.5B with hooks, port 8765)
#   5. Start the controller (FastAPI on port 8001) which spawns llama-server
#      on port 8080
#
# Open http://localhost:8001/ for the dashboard.
#
# Environment knobs:
#   WITH_CUDA=ON        Build/run llama.cpp with CUDA support (slower build)
#   CONTROLLER_PORT=8001 Override controller port (default 8001, since 8000
#                       is often taken by other tooling)
#   N_GPU_LAYERS=99     Number of INT4 model layers to offload to GPU
#                       (set to 0 for pure CPU)
#   CTX=2048            Context window per slot
#   MAX_SESSIONS=4      Max concurrent KV-cached sessions

set -uo pipefail

cd "$(dirname "$0")"
ROOT="$PWD"

mkdir -p logs

echo "==> Glass Box startup"
echo "    $(date)"
echo "    free RAM: $(free -m | awk '/^Mem:/ {print $7 " MB available"}')"
echo "    free disk: $(df -h "$ROOT" | awk 'NR==2 {print $4}')"
echo "    WITH_CUDA=${WITH_CUDA:-OFF}, CONTROLLER_PORT=${CONTROLLER_PORT:-8001}"

# ------------------------------------------------------------------
# 1. Build llama.cpp if needed
# ------------------------------------------------------------------
if [ ! -x "llama.cpp/build/bin/llama-server" ]; then
  echo "==> llama-server not built; building (CPU, ~5 min)..."
  cmake -B llama.cpp/build \
        -DGGML_CUDA=${WITH_CUDA:-OFF} \
        -DGGML_OPENBLAS=${WITH_OPENBLAS:-ON} \
        -DCMAKE_BUILD_TYPE=Release
  cmake --build llama.cpp/build --config Release -j$(nproc) \
        --target llama-server llama-quantize
fi

# ------------------------------------------------------------------
# 2. Quantize the model if needed
# ------------------------------------------------------------------
GGUF_PATH=$(ls models/gguf/*Q4_K_M.gguf models/gguf/*q4_k_m.gguf 2>/dev/null | head -1)
if [ -z "$GGUF_PATH" ]; then
  echo "==> Quantized GGUF not found; running quantization..."
  python3 src/quantize.py --model Qwen/Qwen2.5-1.5B-Instruct --quant Q4_K_M
  GGUF_PATH=$(ls models/gguf/*Q4_K_M.gguf 2>/dev/null | head -1)
fi
if [ -z "$GGUF_PATH" ]; then
  echo "ERROR: no Q4_K_M GGUF found in models/gguf/" >&2
  exit 1
fi
echo "==> Using model: $GGUF_PATH"

# ------------------------------------------------------------------
# 3. Neuron clusters if needed
# ------------------------------------------------------------------
if [ ! -f "models/neuron_clusters.json" ]; then
  echo "==> Computing neuron clusters (PCA calibration pass)..."
  python3 src/neuron_clusters.py --model Qwen/Qwen2.5-0.5B-Instruct
fi

# ------------------------------------------------------------------
# 4. Viz model (background)
# ------------------------------------------------------------------
echo "==> Starting viz model on :8765..."
python3 src/viz_model.py \
  --model Qwen/Qwen2.5-0.5B-Instruct \
  --clusters "$ROOT/models/neuron_clusters.json" \
  --port 8765 \
  > "$ROOT/logs/viz.log" 2>&1 &
VIZ_PID=$!

# ------------------------------------------------------------------
# 5. Controller (background) — spawns its own llama-server
# ------------------------------------------------------------------
echo "==> Starting controller on :${CONTROLLER_PORT:-8001}..."
python3 src/server.py \
  --model "$GGUF_PATH" \
  --llama-port 8080 \
  --controller-port "${CONTROLLER_PORT:-8001}" \
  --n-gpu-layers ${N_GPU_LAYERS:-99} \
  --ctx ${CTX:-2048} \
  --max-sessions ${MAX_SESSIONS:-4} \
  > "$ROOT/logs/controller.log" 2>&1 &
CTRL_PID=$!

# Cleanup handler
cleanup() {
  echo ""
  echo "==> Shutting down..."
  kill $CTRL_PID 2>/dev/null || true
  kill $VIZ_PID 2>/dev/null || true
  sleep 1
  exit 0
}
trap cleanup INT TERM

echo ""
echo "==> Glass Box is running"
echo "    Dashboard:    http://localhost:${CONTROLLER_PORT:-8001}/"
echo "    llama-server: http://localhost:8080/"
echo "    Viz model:    ws://localhost:8765/ws/viz"
echo ""
echo "    Controller PID=$CTRL_PID, log: logs/controller.log"
echo "    Viz model PID=$VIZ_PID, log: logs/viz.log"
echo "    llama-server log: logs/llama-server.log"
echo ""
echo "    Press Ctrl+C to stop"

# Wait for either to die
wait -n $CTRL_PID $VIZ_PID
echo ""
echo "==> A child process exited; shutting down."
cleanup
