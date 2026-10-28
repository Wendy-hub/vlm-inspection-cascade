#!/usr/bin/env bash
# One-time setup. Run each step in its own tmux window:
#   bash scripts/server_setup.sh venv    # venv with anomalib 2.7.0 (+ torch 2.8 unless BASE_PY's env has it)
#   bash scripts/server_setup.sh visa    # VisA dataset, 1.93 GB, official Amazon S3 link
#   bash scripts/server_setup.sh qwen    # Qwen2.5-VL-7B-Instruct, 16.6 GB, from ModelScope
#   bash scripts/server_setup.sh wrn50   # torchvision WideResNet-50 ImageNet weights, 138 MB
set -euo pipefail
source "$(dirname "$0")/env.sh"

case "${1:-}" in
  venv)
    # --system-site-packages: torch / vLLM already in BASE_PY's env are reused, not downloaded again.
    [ -x "$VENV/bin/python" ] || "$BASE_PY" -m venv --system-site-packages "$VENV"
    "$VENV/bin/pip" install -i "$PIP_INDEX" \
      "torch==2.8.0" "torchvision==0.23.0" "anomalib==2.7.0" openai pandas matplotlib scikit-learn
    "$VENV/bin/python" -c "import torch, anomalib; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'anomalib', anomalib.__version__)"
    ;;
  visa)
    mkdir -p "$PROJ_DIR/datasets/visa"
    curl -L -C - --retry 5 --retry-delay 5 -o "$PROJ_DIR/datasets/visa/VisA_20220922.tar" \
      https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar
    ;;
  qwen)
    REPO=Qwen/Qwen2.5-VL-7B-Instruct
    mkdir -p "$MODEL_DIR"
    FILES=$(curl -s "https://www.modelscope.cn/api/v1/models/$REPO/repo/files?Revision=master&Recursive=true" \
      | python3 -c "import sys,json; print('\n'.join(f['Path'] for f in json.load(sys.stdin)['Data']['Files'] if f['Type']=='blob'))")
    for f in $FILES; do
      echo "== $f"
      curl -L -C - --retry 5 --retry-delay 5 -s -S -o "$MODEL_DIR/$f" \
        "https://www.modelscope.cn/models/$REPO/resolve/master/$f"
    done
    du -sh "$MODEL_DIR"
    ;;
  wrn50)
    mkdir -p "$(dirname "$WRN50_WEIGHTS")"
    curl -L -C - --retry 5 -o "$WRN50_WEIGHTS" https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth
    sha256sum "$WRN50_WEIGHTS" | cut -c1-8   # should print 95faca4d
    ;;
  *)
    echo "usage: $0 {venv|visa|qwen|wrn50}"; exit 1 ;;
esac
echo "DONE: $1"
