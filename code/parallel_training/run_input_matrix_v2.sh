#!/usr/bin/env bash
# 输入消融（worker 安全版）：通过 sitecustomize + ABLATE_MODE 在每个进程内生效。
set -uo pipefail

CODE=/g3240101584lt/srtp-ocean/code
SUITE=/g3240101584lt/srtp-ocean/runs/ablation_input_v2_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$SUITE" || exit 1

export PYTHONPATH="$CODE/parallel_training/ablate_site:$CODE/python_pkgs:$CODE:$CODE/parallel_training"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$CODE" || exit 1

printf 'SUITE=%s\nstart=%s\n' "$SUITE" "$(date -u +%FT%TZ)" | tee "$SUITE/suite.log"

MODES="no_smap no_wind no_adt no_context oisst_only adt_only wind_only smap_only no_surface offsets_only"
MAX_PARALLEL=${MAX_PARALLEL:-5}
running=0

run_one() {
  local mode="$1"
  ABLATE_MODE="$mode" /opt/conda/bin/python -u parallel_training/train_parallel.py \
    --dataset-dir /g3240101584lt/srtp-ocean/data/training_ready_v2 \
    --run-dir "$SUITE/$mode" \
    --device cuda --workers 1 --validation-workers 1 --cpu-threads 1 \
    --amp --cache-gib 8 --epochs 200 --patience 201 --seed 20261120 \
    > "$SUITE/$mode.log" 2>&1
  local code=$?
  echo "$code" > "$SUITE/$mode.exitcode"
  printf 'done %s exit=%s %s\n' "$mode" "$code" "$(date -u +%FT%TZ)" >> "$SUITE/suite.log"
}

for mode in $MODES; do
  while (( running >= MAX_PARALLEL )); do
    wait -n || true
    running=$((running - 1))
  done
  run_one "$mode" &
  running=$((running + 1))
  printf 'launched %s\n' "$mode" | tee -a "$SUITE/suite.log"
done
wait || true
printf 'SUITE_DONE=%s\nfinish=%s\n' "$SUITE" "$(date -u +%FT%TZ)" | tee -a "$SUITE/suite.log"
