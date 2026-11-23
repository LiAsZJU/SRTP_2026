#!/usr/bin/env bash
# v4（方案 A）四模型统一训练：Swin / 残差CNN / U-Net / ViT，各 200 轮。
# 输入：training_ready_v4（训练 2005-01–2019-11，验证 2020–2021，测试 2022–2024，前向 2025H1）
# 输入规格：统一无 SMAP（ABLATE_MODE=no_smap；2015 前本来就是全零占位，保证年代一致）
set -uo pipefail

CODE=/g3240101584lt/srtp-ocean/code
DATA=/g3240101584lt/srtp-ocean/data/training_ready_v4
SUITE="/g3240101584lt/srtp-ocean/runs/v4A_four_models_seed20261122_20261122T053535Z"
test -d "$SUITE" || exit 1
test ! -e "$SUITE/suite.log" || exit 1

export PYTHONPATH="$CODE/parallel_training/ablate_site:$CODE/python_pkgs:$CODE:$CODE/parallel_training"
export ABLATE_MODE=no_smap
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$CODE" || exit 1

printf 'SUITE=%s\nstart=%s\n' "$SUITE" "$(date -u +%FT%TZ)" | tee "$SUITE/suite.log"

COMMON=(--dataset-dir "$DATA" --device cuda --workers 2 --validation-workers 1 --cpu-threads 1
        --amp --cache-gib 14 --epochs 200 --patience 201 --seed 20261122)

run_model() {
  local arch="$1"; shift
  /opt/conda/bin/python -u parallel_training/train_arch.py --arch "$arch" "$@" > "$SUITE/$arch.log" 2>&1
  local code=$?
  echo "$code" > "$SUITE/$arch.exitcode"
  printf 'done %s exit=%s %s\n' "$arch" "$code" "$(date -u +%FT%TZ)" >> "$SUITE/suite.log"
}

run_model swin --run-dir "$SUITE/swin" "${COMMON[@]}" --dim 48 --blocks 4 --heads 4 --window 8 &
run_model cnn  --run-dir "$SUITE/cnn"  "${COMMON[@]}" --dim 48 --blocks 4 &
run_model unet --run-dir "$SUITE/unet" "${COMMON[@]}" --dim 24 &
run_model vit  --run-dir "$SUITE/vit"  "${COMMON[@]}" --dim 96 --blocks 4 --heads 4 &
wait || true

printf 'SUITE_DONE=%s\nfinish=%s\n' "$SUITE" "$(date -u +%FT%TZ)" | tee -a "$SUITE/suite.log"
