#!/usr/bin/env bash
# Shared foreground launcher. No CPU training fallback is allowed.
set -euo pipefail
model="$1"
shift
for argument in "$@"; do
    case "$argument" in
        --run-dir|--run-dir=*|--dataset-dir|--dataset-dir=*|--device|--device=*|--seed|--seed=*)
            printf 'Use RUN_DIR, DATASET_DIR or SEED environment variables; device is fixed to CUDA.\n' >&2
            exit 1 ;;
    esac
done
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
project_dir="$(cd -- "$code_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-/opt/conda/bin/python}"
export PYTHONPATH="$code_dir/python_pkgs:$code_dir:$code_dir/dataset_v2${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8
seed="${SEED:-20261117}"
export PYTHONHASHSEED="$seed"
dataset_dir="${DATASET_DIR:-$project_dir/data/training_ready_v2}"
run_dir="${RUN_DIR:-$project_dir/runs/${model}_v2_seed${seed}_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
if [[ -e "$run_dir" || -e "${run_dir}.log" ]]; then
    printf 'Refusing existing output: %s\n' "$run_dir" >&2
    exit 1
fi
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA required; CPU training is disabled"'
mkdir -p -- "$(dirname -- "$run_dir")"
arguments=(--run-dir "$run_dir" --dataset-dir "$dataset_dir" --device cuda --seed "$seed"
    --workers "${WORKERS:-6}" --validation-workers "${VALIDATION_WORKERS:-2}"
    --cpu-threads 1 --cache-gib "${CACHE_GIB:-8}")
case "$model" in
    residual_cnn|unet_baseline|vit_baseline)
        arguments+=(--epochs "${EPOCHS:-200}" --patience "${PATIENCE:-500}") ;;
    climatology_baseline|eof_ridge) ;;
    *) printf 'Unknown model: %s\n' "$model" >&2; exit 1 ;;
esac
"$python_bin" -u -m "$model.train" "${arguments[@]}" "$@" 2>&1 | tee "${run_dir}.log"
"$python_bin" -u -m climatology_baseline.evaluate \
    --dataset-dir "$dataset_dir" --checkpoint "$run_dir/best.pt" \
    --output "$run_dir/test_metrics.json" --splits test_locked forward_test \
    --device cuda --cpu-threads 1 2>&1 | tee -a "${run_dir}.log"
printf 'Completed: %s\n' "$run_dir"
