#!/bin/bash
# Start the annotation service under CUDA MPS, detached from this SSH session. Usage: serve.sh ROOT PROCESSES
set -euo pipefail
ROOT=$1
PROCESSES=$2
export CUDA_MPS_PIPE_DIRECTORY=/tmp/backcast-mps/pipe CUDA_MPS_LOG_DIRECTORY=/tmp/backcast-mps/log
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY" "$ROOT/logs"
# Start the daemon unless one is already answering; a service restart reuses it.
echo get_server_list | nvidia-cuda-mps-control >/dev/null 2>&1 || nvidia-cuda-mps-control -d
# MPS clients see only the server's device, so the pod's one GPU is device 0.
export CUDA_VISIBLE_DEVICES=0
# PyTorch caches freed GPU blocks up to its limit, so the limit decides how much of the card a process holds. Each
# process gets its share, at most 7 GiB, leaving room for its CUDA context. Live use is only about 1 GiB per process.
TOTAL_MIB=$(nvidia-smi --id=0 --query-gpu=memory.total --format=csv,noheader,nounits)
ALLOCATOR_GIB=$(awk -v mib="$TOTAL_MIB" -v n="$PROCESSES" 'BEGIN { g = (mib * 0.90 / n - 400) / 1024; printf "%.2f", (g > 7 ? 7 : g) }')
if ! awk -v g="$ALLOCATOR_GIB" 'BEGIN { exit !(g >= 1.6) }'; then
  echo "$PROCESSES processes do not fit on this ${TOTAL_MIB} MiB GPU (${ALLOCATOR_GIB} GiB each, 1.6 needed); lower processes in fleet.toml" >&2
  exit 1
fi
export TORCHINDUCTOR_CACHE_DIR="$ROOT/compile_cache/inductor" TRITON_CACHE_DIR="$ROOT/compile_cache/triton" TORCHINDUCTOR_COMPILE_THREADS=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export YOLO_AUTOINSTALL=false YOLO_CONFIG_DIR=/tmp/ultralytics MPLCONFIGDIR=/tmp/matplotlib
rm -f "$ROOT/queue/STOP"
# The pipeline measured in docs/speed_test.md: exact GPU crops and waits, float16 Hydra weights, compiled Hydra and GRU.
nohup setsid python3 -m deepstarnoodles_inference serve-videos --queue "$ROOT/queue" --output "$ROOT/out" \
  --models "$ROOT/models/manifest.json" --processes "$PROCESSES" \
  --device cuda:0 --detector-batch-size 1 --crop-batch-size 1 --allocator-limit-gib "$ALLOCATOR_GIB" --ram-limit-gib 24 \
  --hydra-output-mode minimal --gru-execution graph --gru-check-interval 1 --hydra-execution graph \
  --detect-ahead 2 --decode-prefetch-frames 4 --gpu-crops --cuda-blocking-sync --hydra-fp16-weights \
  --hydra-compile --gru-compile --gru-graph-first-step >> "$ROOT/logs/service.log" 2>&1 < /dev/null &
echo $! > "$ROOT/service.pid"
