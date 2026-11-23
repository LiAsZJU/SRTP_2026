#!/usr/bin/env bash
set -euo pipefail

ROOT=/g3240101584lt/srtp-ocean
RUN_DIR="$ROOT/runs/swin_physics_v1"
mkdir -p "$RUN_DIR"
export PYTHONPATH="$ROOT/code/python_pkgs:$ROOT/code"

if ! /opt/conda/bin/python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
  echo "No CUDA GPU is visible; training was not started." >&2
  exit 1
fi

if [[ -f "$RUN_DIR/pid" ]] && kill -0 "$(cat "$RUN_DIR/pid")" 2>/dev/null; then
  echo "already running pid=$(cat "$RUN_DIR/pid")"
  exit 0
fi

nohup /opt/conda/bin/python -u "$ROOT/code/train_swin_physics.py" \
  --run-dir "$RUN_DIR" --device cuda --amp --resume \
  > "$RUN_DIR/train.log" 2>&1 &
echo $! > "$RUN_DIR/pid"
echo "started pid=$(cat "$RUN_DIR/pid") log=$RUN_DIR/train.log"
