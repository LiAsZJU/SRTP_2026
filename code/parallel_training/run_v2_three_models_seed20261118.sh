#!/usr/bin/env bash
# Three fresh v2 CUDA runs with seed 20261118; preserve existing experiments.
set -euo pipefail
seed=20261118
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
project_dir="$(cd -- "$code_dir/.." && pwd)"
export PYTHONPATH="$code_dir/python_pkgs:$code_dir:$code_dir/dataset_v2:$code_dir/parallel_training"
unset ABLATE_MODE
/opt/conda/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA required; CPU training is disabled"'
test -r "$project_dir/data/training_ready_v2/training_manifest.json"
suite_dir="${1:-$project_dir/runs/three_models_v2_seed20261118_$(date -u +%Y%m%dT%H%M%SZ)}"
if [[ -e "$suite_dir" ]]; then
    printf 'Refusing existing suite: %s\n' "$suite_dir" >&2
    exit 1
fi
mkdir -- "$suite_dir"
export DATASET_DIR="$project_dir/data/training_ready_v2"
export EPOCHS=200 PATIENCE=201 SEED="$seed"
export WORKERS=6 VALIDATION_WORKERS=2 CPU_THREADS=1 CACHE_GIB=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
ulimit -n 65536
run_one() {
    local model="$1"
    local result=0
    if [[ "$model" == swin ]]; then
        RUN_DIR="$suite_dir/$model" bash "$code_dir/parallel_training/run_task_500ep.sh" >"$suite_dir/$model.driver.log" 2>&1 || result=$?
    else
        RUN_DIR="$suite_dir/$model" bash "$code_dir/$model/run_gpu.sh" >"$suite_dir/$model.driver.log" 2>&1 || result=$?
    fi
    printf '%s\n' "$result" >"$suite_dir/$model.exitcode"
}
printf 'Started %s\n' "$(date -u +%FT%TZ)"
pids=()
for model in swin unet_baseline vit_baseline; do
    run_one "$model" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
printf 'Finished %s\n' "$(date -u +%FT%TZ)"
