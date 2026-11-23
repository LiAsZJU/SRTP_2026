#!/usr/bin/env bash
# 构建 training_ready_v4（方案 A）：先 fit 统计量，再 publish manifest。
set -uo pipefail

CODE=/g3240101584lt/srtp-ocean/code
LOG=/g3240101584lt/srtp-ocean/data/build_v4.log
cd "$CODE" || exit 1
export PYTHONPATH="$CODE/python_pkgs:$CODE"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

{
  echo "START $(date -u +%FT%TZ)"
  /opt/conda/bin/python -u dataset_v2/build_dataset_v4.py fit
  echo "FIT_EXIT=$? $(date -u +%FT%TZ)"
  /opt/conda/bin/python -u dataset_v2/build_dataset_v4.py publish
  echo "PUBLISH_EXIT=$? $(date -u +%FT%TZ)"
  echo "END $(date -u +%FT%TZ)"
} >> "$LOG" 2>&1
