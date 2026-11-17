#!/usr/bin/env bash
# Launch Glass Box Jetson with the verified BotPsy-0.8B configuration:
#   served: BotPsy-0.8B Q4_K_M-imat GGUF (llama-server, CUDA, all layers on GPU)
#   viz:    qvac/BotPsy-0.8B FP16 twin (CUDA venv) + precomputed PCA basis
# Dashboard: http://<jetson-ip>:8001/
set -euo pipefail
cd "$(dirname "$0")/.."

CLUSTERS=models/neuron_clusters_botpsy.json
if [ ! -f "$CLUSTERS" ]; then
  echo "==> one-time PCA calibration for the viz twin..."
  ~/.venvs/jetson-pytorch/bin/python src/neuron_clusters.py \
    --model qvac/BotPsy-0.8B --out "$CLUSTERS"
fi

exec /usr/bin/python3.10 src/server.py \
  --model ~/models/new-zoo/botpsy-0.8b-q4_k_m-imat.gguf \
  --llama-bin ~/llama.cpp/build/bin/llama-server \
  --llama-port 8080 \
  --controller-port 8001 \
  --n-gpu-layers 99 \
  --ctx 2048 \
  --max-sessions 4 \
  --viz-model qvac/BotPsy-0.8B \
  --viz-python ~/.venvs/jetson-pytorch/bin/python \
  --viz-clusters "$CLUSTERS" \
  --temperature 0.7
