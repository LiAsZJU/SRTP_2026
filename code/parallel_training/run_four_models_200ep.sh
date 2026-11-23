#!/usr/bin/env bash
# One fresh 200-epoch run per model; existing experiments are never reused.
set -euo pipefail
suite_dir="$1"
seed=20261120
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
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
for model in residual_cnn unet_baseline vit_baseline swin; do
    run_one "$model" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
printf 'Finished %s\n' "$(date -u +%FT%TZ)"
