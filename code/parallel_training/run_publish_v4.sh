#!/usr/bin/env bash
# 仅执行 v4 的 publish 阶段（fit 已完成），完成后自动启动四模型训练。
set -uo pipefail

CODE=/g3240101584lt/srtp-ocean/code
LOG=/g3240101584lt/srtp-ocean/data/build_v4_publish.log
cd "$CODE" || exit 1
export PYTHONPATH="$CODE/python_pkgs:$CODE"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4

{
  echo "START $(date -u +%FT%TZ)"
  /opt/conda/bin/python -u dataset_v2/build_dataset_v4.py publish
  code=$?
  echo "PUBLISH_EXIT=$code $(date -u +%FT%TZ)"
  echo "END $(date -u +%FT%TZ)"
} >> "$LOG" 2>&1
