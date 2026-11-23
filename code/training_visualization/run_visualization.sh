#!/usr/bin/env bash
# Read existing reports only. Does not import any training package or CUDA API.
set -euo pipefail
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$code_dir${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg
"${PYTHON_BIN:-/opt/conda/bin/python}" -m training_visualization "$@"
