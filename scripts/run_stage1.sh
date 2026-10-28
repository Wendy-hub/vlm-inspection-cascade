#!/usr/bin/env bash
# Run stage 1 (PatchCore).
# Usage (in tmux): GPU=0 bash scripts/run_stage1.sh [--prepare-only | --categories cat1 cat2 ...]
set -euo pipefail
source "$(dirname "$0")/env.sh"
export CUDA_VISIBLE_DEVICES="${GPU:-0}"
cd "$PROJ_DIR"
exec "$VENV/bin/python" src/stage1_patchcore.py "$@"
