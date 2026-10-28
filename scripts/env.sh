#!/usr/bin/env bash
# Paths and settings shared by the scripts. Sourced, not run.
# Machine-specific values go in scripts/env.local.sh (gitignored); copy scripts/env.example.sh to start.

PROJ_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -f "$PROJ_DIR/scripts/env.local.sh" ]; then
  # shellcheck source=/dev/null
  source "$PROJ_DIR/scripts/env.local.sh"
fi

: "${VENV:=$PROJ_DIR/.venv}"                     # venv with anomalib (scripts/server_setup.sh venv)
: "${BASE_PY:=python3}"                           # interpreter used to create that venv
: "${VLLM_PY:=$VENV/bin/python}"                  # any python with vLLM installed
: "${MODEL_DIR:=$PROJ_DIR/models/Qwen2.5-VL-7B-Instruct}"
: "${WRN50_WEIGHTS:=$PROJ_DIR/models/torchvision/wide_resnet50_2-95faca4d.pth}"
: "${PIP_INDEX:=https://pypi.org/simple}"
export WRN50_WEIGHTS

# Some shared machines carry a stale proxy in their environment; set CLEAR_PROXY=1 to drop it.
if [ "${CLEAR_PROXY:-0}" = 1 ]; then
  unset all_proxy http_proxy https_proxy ALL_PROXY HTTP_PROXY HTTPS_PROXY no_proxy NO_PROXY
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID    # GPU numbers match nvidia-smi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}" MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
