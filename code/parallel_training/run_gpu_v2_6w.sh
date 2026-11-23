#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/../.." && pwd)"
dataset_dir="$project_dir/data/training_ready_v2"
seed="${SEED:-20261117}"
export PYTHONHASHSEED="$seed"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

if [[ ! -r "$dataset_dir/training_manifest.json" ]]; then
  printf 'Cannot read the v2 training manifest in %s\n' "$dataset_dir" >&2
  exit 1
fi

if ! nvidia-smi --query-gpu=name --format=csv,noheader | grep 'A800' >/dev/null; then
  printf 'Start this script in an AiStation environment with an A800 GPU.\n' >&2
  exit 1
fi

run_dir="$project_dir/runs/swin_physics_parallel_data_v2_gpu_6w_$(date -u +%Y%m%dT%H%M%SZ)_$$"
log_file="$run_dir.log"
printf 'Run directory: %s\nLog: %s\n' "$run_dir" "$log_file"

bash "$script_dir/run_parallel.sh" \
  --dataset-dir "$dataset_dir" \
  --run-dir "$run_dir" \
  --device cuda \
  --seed "$seed" \
  --workers 6 --validation-workers 2 --cpu-threads 1 \
  --amp --cache-gib 8 \
  --epochs 200 --patience 8 \
  2>&1 | tee "$log_file"

printf '\nEvaluating the best checkpoint on the 2024 main and 2025 Jan-Jun supplementary test splits.\n' | tee -a "$log_file"
code_dir="$project_dir/code"
export PYTHONPATH="$code_dir/python_pkgs:$code_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

/opt/conda/bin/python -u "$code_dir/dataset_v2/evaluate_checkpoint.py" \
  --dataset-dir "$dataset_dir" \
  --checkpoint "$run_dir/best.pt" \
  --output "$run_dir/test_metrics.json" \
  --splits test_locked forward_test --device cuda --cpu-threads 1 \
  2>&1 | tee -a "$log_file"
