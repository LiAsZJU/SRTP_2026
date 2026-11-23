#!/usr/bin/env bash
set -uo pipefail
cd /g3240101584lt/srtp-ocean/code
export PYTHONPATH=/g3240101584lt/srtp-ocean/code/python_pkgs:/g3240101584lt/srtp-ocean/code
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
SUITE=/g3240101584lt/srtp-ocean/runs/ablation_input_$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p "$SUITE"
echo "SUITE=$SUITE" > "$SUITE/suite.log"
echo "start $(date -u +%FT%TZ)" >> "$SUITE/suite.log"
COMMON="--dataset-dir /g3240101584lt/srtp-ocean/data/training_ready_v2 --device cuda --workers 1 --validation-workers 1 --cpu-threads 1 --amp --cache-gib 8 --epochs 200 --patience 201 --seed 20261120"
run() {
  local m="$1"
  /opt/conda/bin/python -u parallel_training/ablate_input.py --ablate "$m" $COMMON --run-dir "$SUITE/$m" > "$SUITE/$m.log" 2>&1
  local code=$?
  echo "$code" > "$SUITE/$m.exitcode"
  echo "done $m exit=$code $(date -u +%FT%TZ)" >> "$SUITE/suite.log"
}
for m in no_surface offsets_only adt_only wind_only smap_only; do run "$m" & done
wait
echo "SUITE_DONE=$SUITE" >> "$SUITE/suite.log"
echo "finish $(date -u +%FT%TZ)" >> "$SUITE/suite.log"
