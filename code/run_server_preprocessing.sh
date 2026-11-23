#!/usr/bin/env bash
set -euo pipefail

ROOT=/g3240101584lt/srtp-ocean
RUN_DIR="$ROOT/runs/training_ready_v1"
mkdir -p "$RUN_DIR"

if [[ -f "$RUN_DIR/pid" ]] && kill -0 "$(cat "$RUN_DIR/pid")" 2>/dev/null; then
  echo "already running pid=$(cat "$RUN_DIR/pid")"
  exit 0
fi

export PYTHONPATH="$ROOT/code/python_pkgs"
nohup /opt/conda/bin/python -u "$ROOT/code/prepare_server_training_data.py" \
  --stages cora argo statistics manifest validate \
  > "$RUN_DIR/preprocessing.log" 2>&1 &
echo $! > "$RUN_DIR/pid"
echo "started pid=$(cat "$RUN_DIR/pid") log=$RUN_DIR/preprocessing.log"
