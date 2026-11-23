#!/usr/bin/env bash
# Read-only spatial/depth diagnostics and 2D ocean CT/SA slicing runner.
# Supports both CPU and CUDA (--device auto|cpu|cuda). Never starts training.
set -euo pipefail
code_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$code_dir:$code_dir/dataset_v2:$code_dir/python_pkgs${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export MPLBACKEND=Agg
"${PYTHON_BIN:-/opt/conda/bin/python}" -m spatial_depth_diagnostics "$@"
