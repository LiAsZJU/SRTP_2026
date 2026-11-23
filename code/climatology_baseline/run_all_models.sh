#!/usr/bin/env bash
# Fit the two references, then train the three seeded neural baselines.
set -euo pipefail
suite_dir="$1"
phase="${2:-all}"
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export EPOCHS=200 WORKERS=6 VALIDATION_WORKERS=2 CACHE_GIB=8
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
ulimit -n 65536
run_reference() {
    local model="$1"
    local result=0
    RUN_DIR="$suite_dir/$model" bash "$code_dir/$model/run_gpu.sh" >"$suite_dir/$model.driver.log" 2>&1 || result=$?
    printf '%s\n' "$result" >"$suite_dir/$model.exitcode"
    return 0
}
run_neural() {
    local model="$1"
    local result=0
    GROUP_DIR="$suite_dir/$model" bash "$code_dir/$model/run_seeds.sh" >"$suite_dir/$model.driver.log" 2>&1 || result=$?
    printf '%s\n' "$result" >"$suite_dir/$model.exitcode"
    return 0
}
printf 'Suite started at %s\n' "$(date -u +%FT%TZ)"
run_reference climatology_baseline
run_reference eof_ridge
if [[ "$phase" == "references" ]]; then
    exit 0
fi
pids=()
for model in residual_cnn unet_baseline vit_baseline; do
    run_neural "$model" &
    pids+=("$!")
done
for pid in "${pids[@]}"; do
    wait "$pid"
done
printf 'Suite finished at %s\n' "$(date -u +%FT%TZ)"
