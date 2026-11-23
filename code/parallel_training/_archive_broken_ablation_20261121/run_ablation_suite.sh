#!/usr/bin/env bash
# 消融套件：物理约束 λ 扫描 + 输入源消融（复用 train_parallel 的 Model/Objective，仅用 wrapper 裁剪通道）
# 用法：bash run_ablation_suite.sh /g3240101584lt/srtp-ocean/runs/ablation_YYYYmmdd
set -euo pipefail

BASE=/g3240101584lt/srtp-ocean
CODE=$BASE/code
SUITE=${1:?用法: bash run_ablation_suite.sh <suite_dir>}
mkdir -p "$SUITE"

export PYTHONPATH=$CODE/python_pkgs:$CODE
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1

COMMON=(--dataset-dir "$BASE/data/training_ready_v2" --device cuda
        --workers 2 --validation-workers 1 --cpu-threads 1
        --amp --cache-gib 8 --epochs 200 --patience 201 --seed 20261120)

# 任务列表： 名称 | 消融模式 | 追加参数
JOBS=(
  "A1_lambda0|full|--lambda-physics 0"
  "A1_lambda0p1|full|--lambda-physics 0.1"
  "A1_lambda1p0|full|--lambda-physics 1.0"
  "B1_no_smap|no_smap|"
  "B1_no_wind|no_wind|"
  "B1_no_adt|no_adt|"
  "B1_oisst_only|oisst_only|"
  "B1_no_context|no_context|"
)
MAX_PARALLEL=${MAX_PARALLEL:-4}

launch() {
  local name="$1" mode="$2" extra="$3"
  ( cd "$CODE" && /opt/conda/bin/python -u parallel_training/ablate_input.py \
      --ablate "$mode" "${COMMON[@]}" --run-dir "$SUITE/$name" $extra \
      > "$SUITE/$name.log" 2>&1 ; echo "$?" > "$SUITE/$name.exitcode" ) &
}

printf 'suite=%s  start=%s\n' "$SUITE" "$(date -u +%FT%TZ)" | tee "$SUITE/suite.log"
running=0
for job in "${JOBS[@]}"; do
  IFS='|' read -r name mode extra <<< "$job"
  while (( running >= MAX_PARALLEL )); do
    wait -n || true
    running=$((running - 1))
  done
  launch "$name" "$mode" "$extra"
  running=$((running + 1))
  printf 'launched %s (mode=%s)\n' "$name" "$mode" | tee -a "$SUITE/suite.log"
done
wait || true
printf 'suite finished %s\n' "$(date -u +%FT%TZ)" | tee -a "$SUITE/suite.log"
ls -1 "$SUITE"/*.exitcode 2>/dev/null | while read -r f; do printf '%s=%s\n' "$(basename "$f" .exitcode)" "$(cat "$f")"; done | tee -a "$SUITE/suite.log"
