#!/usr/bin/env bash
# Run a project script with the project venv.
# Usage: bash scripts/run_py.sh src/build_band.py [args...]
set -euo pipefail
source "$(dirname "$0")/env.sh"
export PYTHONWARNINGS=ignore
cd "$PROJ_DIR"
exec "$VENV/bin/python" "$@"
