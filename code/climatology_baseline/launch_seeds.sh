#!/usr/bin/env bash
# Run one architecture sequentially over distinct random seeds.
set -euo pipefail
model="$1"
shift
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
project_dir="$(cd -- "$code_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-/opt/conda/bin/python}"
export PYTHONPATH="$code_dir/python_pkgs:$code_dir:$code_dir/dataset_v2${PYTHONPATH:+:$PYTHONPATH}"
if [[ "$#" -eq 0 ]]; then
    set -- 20261117 20261118 20261119
fi
group_dir="${GROUP_DIR:-$project_dir/runs/${model}_v2_seeds_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA required; CPU training is disabled"'
mkdir -- "$group_dir"
run_dirs=()
for seed in "$@"; do
    run_dir="$group_dir/seed_$seed"
    SEED="$seed" RUN_DIR="$run_dir" bash "$code_dir/$model/run_gpu.sh"
    run_dirs+=("$run_dir")
done
"$python_bin" -m climatology_baseline.summarize_seeds \
    --runs "${run_dirs[@]}" --output "$group_dir/seed_summary.json"
printf 'Seed summary: %s/seed_summary.json\n' "$group_dir"
