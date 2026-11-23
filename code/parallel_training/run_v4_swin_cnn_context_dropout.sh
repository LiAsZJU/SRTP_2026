#!/usr/bin/env bash
# v4（no_smap）Swin + 残差 CNN 训练期 Argo C Context Dropout 并行启动脚本。
# 默认：training_ready_v4，200 轮，patience=201，mixed Context Dropout，训练后自动运行
# full-context 与 zero-context 两套测试（均含全部 E 与验证期未见 E 分组）。
set -uo pipefail

CODE=/g3240101584lt/srtp-ocean/code
DATA="${DATASET_DIR:-/g3240101584lt/srtp-ocean/data/training_ready_v4}"
SEED="${SEED:-20261123}"
EPOCHS="${EPOCHS:-200}"
PATIENCE="${PATIENCE:-201}"
MODE="${CONTEXT_DROP_MODE:-mixed}"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
SUITE="${SUITE_DIR:-/g3240101584lt/srtp-ocean/runs/v4A_swin_cnn_ctxdrop_${MODE}_seed${SEED}_${TS}}"

if [[ -e "$SUITE" ]]; then
  printf 'Refusing existing suite directory: %s\n' "$SUITE" >&2
  exit 1
fi

/opt/conda/bin/python -c 'import torch; assert torch.cuda.is_available(), "CUDA required; CPU training is disabled"'
test -r "$DATA/training_manifest.json" || exit 1

export PYTHONPATH="$CODE/parallel_training/ablate_site:$CODE/python_pkgs:$CODE:$CODE/parallel_training"
export ABLATE_MODE=no_smap
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
cd "$CODE" || exit 1

mkdir -p "$SUITE"
printf 'SUITE=%s\nseed=%s\nmode=%s\nstart=%s\n' "$SUITE" "$SEED" "$MODE" "$(date -u +%FT%TZ)" | tee "$SUITE/suite.log"

COMMON=(
  --dataset-dir "$DATA"
  --device cuda
  --workers 2
  --validation-workers 1
  --cpu-threads 1
  --amp
  --cache-gib 14
  --epochs "$EPOCHS"
  --patience "$PATIENCE"
  --seed "$SEED"
  --context-drop-mode "$MODE"
  --context-drop-p-clean "${CONTEXT_DROP_P_CLEAN:-0.15}"
  --context-drop-p-full "${CONTEXT_DROP_P_FULL:-0.20}"
  --context-drop-p-block "${CONTEXT_DROP_P_BLOCK:-0.35}"
  --context-drop-min-keep "${CONTEXT_DROP_MIN_KEEP:-0.10}"
  --context-drop-max-keep "${CONTEXT_DROP_MAX_KEEP:-0.85}"
  --require-no-smap
  --evaluate
  --eval-no-context
)

run_model() {
  local arch="$1"; shift
  /opt/conda/bin/python -u parallel_training/train_v4_context_dropout.py --arch "$arch" "$@" > "$SUITE/$arch.log" 2>&1
  local code=$?
  echo "$code" > "$SUITE/$arch.exitcode"
  printf 'done %s exit=%s %s\n' "$arch" "$code" "$(date -u +%FT%TZ)" >> "$SUITE/suite.log"
}

run_model swin --run-dir "$SUITE/swin" "${COMMON[@]}" --dim 48 --blocks 4 --heads 4 --window 8 &
run_model cnn  --run-dir "$SUITE/cnn"  "${COMMON[@]}" --dim 48 --blocks 4 &
wait || true

printf 'SUITE_DONE=%s\nfinish=%s\n' "$SUITE" "$(date -u +%FT%TZ)" | tee -a "$SUITE/suite.log"
