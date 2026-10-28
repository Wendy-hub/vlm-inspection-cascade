#!/usr/bin/env bash
# Serve Qwen2.5-VL-7B with vLLM (tested with vLLM 0.11). Uses $VLLM_PY, so vLLM can live in another env.
# The server needs ~24 GB in total: GPU_UTIL=0.5 on a 48 GB card, 0.35 on 80 GB.
# Usage (in tmux): CUDA_VISIBLE_DEVICES=0 [PORT=8901] [GPU_UTIL=0.5] bash scripts/serve_vlm.sh
set -euo pipefail
source "$(dirname "$0")/env.sh"
export HF_HUB_OFFLINE=1 VLLM_NO_USAGE_STATS=1 DO_NOT_TRACK=1

# 448x448 max per image (~256 visual tokens); up to 6 images per request.
exec "$VLLM_PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL_DIR" --served-model-name qwen2.5-vl-7b \
  --host 127.0.0.1 --port "${PORT:-8901}" \
  --gpu-memory-utilization "${GPU_UTIL:-0.5}" --max-model-len 8192 --max-num-seqs 32 \
  --limit-mm-per-prompt '{"image": 6}' \
  --mm-processor-kwargs '{"max_pixels": 200704}'
