#!/usr/bin/env bash
# AiStation 任务管理入口脚本（默认最多 200 轮 GPU 训练，文件名保留历史命名）
# 说明：
#   - 不覆盖任何已有脚本；原 run_gpu_v2_6w.sh 保持不动
#   - 默认 200 轮、6 训练 worker / 2 验证 worker、1 个 CPU 计算线程、8 GiB 缓存
#   - 训练结束后用 best.pt 在 test_locked / forward_test 上评估
#   - 所有参数都可以用环境变量覆盖，也可以直接在命令行追加参数（追加的参数优先）
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd -- "$script_dir/../.." && pwd)"
dataset_dir="${DATASET_DIR:-$project_dir/data/training_ready_v2}"
device="${DEVICE:-cuda}"
epochs="${EPOCHS:-200}"
patience="${PATIENCE:-8}"
seed="${SEED:-20261117}"
export PYTHONHASHSEED="$seed"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "host=$(hostname) start=$(date -u +%FT%TZ)"
echo "script_dir=$script_dir"
echo "project_dir=$project_dir"
echo "dataset_dir=$dataset_dir"
echo "device=$device epochs=$epochs patience=$patience"
echo "seed=$seed"

if [[ ! -r "$dataset_dir/training_manifest.json" ]]; then
  printf 'Cannot read the v2 training manifest in %s\n' "$dataset_dir" >&2
  exit 1
fi

gpu_name="$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1 || true)"
echo "gpu=${gpu_name:-<none>}"

if [[ "$device" == "cuda" ]]; then
  if [[ -z "$gpu_name" ]]; then
    printf 'No GPU is visible but device=cuda. Aborting.\n' >&2
    exit 1
  fi
  if [[ "${REQUIRE_A800:-0}" == "1" && "$gpu_name" != *A800* ]]; then
    printf 'REQUIRE_A800=1 but the visible GPU is %s. Aborting.\n' "$gpu_name" >&2
    exit 1
  fi
fi

# run_dir 必须不存在（train_parallel.py 用 exist_ok=False 创建），
# 所以这里只创建日志所在目录，绝不预先创建 run_dir。
run_dir="${RUN_DIR:-$project_dir/runs/swin_physics_v2_${epochs}ep_${device}_$(date -u +%Y%m%dT%H%M%SZ)_$$}"
log_file="${LOG_FILE:-$run_dir.log}"
if [[ -e "$run_dir" ]]; then
  printf 'run_dir already exists: %s\n' "$run_dir" >&2
  exit 1
fi
mkdir -p "$(dirname -- "$log_file")"

printf 'run_dir=%s\nlog=%s\n' "$run_dir" "$log_file"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || true

# 训练：前台执行（任务管理里绝对不要加 nohup ... &）
bash "$script_dir/run_parallel.sh" \
  --dataset-dir "$dataset_dir" \
  --run-dir "$run_dir" \
  --device "$device" \
  --seed "$seed" \
  --workers "${WORKERS:-6}" \
  --validation-workers "${VALIDATION_WORKERS:-2}" \
  --cpu-threads "${CPU_THREADS:-1}" \
  --amp \
  --cache-gib "${CACHE_GIB:-8}" \
  --epochs "$epochs" \
  --patience "$patience" \
  "$@" 2>&1 | tee "$log_file"

echo "training finished at $(date -u +%FT%TZ)"

code_dir="$project_dir/code"
export PYTHONPATH="$code_dir/python_pkgs:$code_dir${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1

if [[ -f "$run_dir/best.pt" ]]; then
  printf '\nEvaluating best.pt on test_locked and forward_test.\n' | tee -a "$log_file"
  /opt/conda/bin/python -u "$code_dir/dataset_v2/evaluate_checkpoint.py" \
    --dataset-dir "$dataset_dir" \
    --checkpoint "$run_dir/best.pt" \
    --output "$run_dir/test_metrics.json" \
    --splits test_locked forward_test \
    --device "$device" --cpu-threads "${CPU_THREADS:-1}" \
    2>&1 | tee -a "$log_file" \
    || echo "[WARN] evaluation failed; training artifacts kept in $run_dir" >&2
else
  echo "[WARN] best.pt not found; skip evaluation" >&2
fi

echo "all done at $(date -u +%FT%TZ)"
