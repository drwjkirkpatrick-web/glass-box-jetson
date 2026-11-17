#!/usr/bin/env bash
# Clean rebuild of llama.cpp's CUDA target for Jetson Orin (SM 8.7).
# On Jetson ARM this build takes ~1-2 hours because of the ~120 CUDA
# template instantiations (flash-attention, mmq, mmf). Run in background
# and watch progress separately.
#
# Usage:  ./rebuild_cuda.sh [jobs]
#   jobs: parallelism (default 3 — Jetson has 6 cores; 3 leaves headroom)
#
# After it finishes, restart with:
#   WITH_CUDA=ON ./start.sh

set -euo pipefail

# Resolve repo root from this script's location, so the script works
# regardless of where the repo is checked out.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$SCRIPT_DIR"
BUILD_DIR="$REPO_DIR/llama.cpp/build-cuda"
LOG_FILE="/tmp/glassbox_cuda_build.log"
PID_FILE="/tmp/glassbox_cuda_build.pid"
JOBS="${1:-3}"

echo "[rebuild_cuda] removing stale build dir: $BUILD_DIR"
rm -rf "$BUILD_DIR"

echo "[rebuild_cuda] configuring (CMAKE_CUDA_ARCHITECTURES=87-real)"
cmake -B "$BUILD_DIR" -S "$REPO_DIR/llama.cpp" \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="${CMAKE_CUDA_COMPILER:-/usr/local/cuda/bin/nvcc}" \
  -DCMAKE_CUDA_ARCHITECTURES=87-real \
  -DCMAKE_BUILD_TYPE=Release

echo "[rebuild_cuda] building with -j$JOBS in background, logging to $LOG_FILE"
nohup cmake --build "$BUILD_DIR" --config Release -j"$JOBS" \
  --target llama-server llama-quantize \
  > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

echo "[rebuild_cuda] started, PID=$(cat "$PID_FILE")"
echo "[rebuild_cuda] watch progress with:"
echo "    tail -f $LOG_FILE"
echo "[rebuild_cuda] when done, restart with:  WITH_CUDA=ON $REPO_DIR/start.sh"
