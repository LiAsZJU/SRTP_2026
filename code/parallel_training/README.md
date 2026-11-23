# 按月并行训练

`run_parallel.sh` 启动单 GPU Swin 训练。默认并行读取 4 个训练月或 2 个验证月，保留原模型、样本月份和顺序、损失、验证指标、50 轮上限和早停参数。CPU 下可用 `--device cpu` 验证流程。

## A800 短测

在卡空闲时，用新目录跑两轮。第一轮填充缓存，第二轮检查复用效果。

```bash
cd /workspace/srtp-ocean
bash code/parallel_training/run_parallel.sh \
  --run-dir runs/swin_physics_parallel_bench_v1 \
  --epochs 2 --max-train-months 12 --max-validation-months 4 --cache-gib 8
```

查看 `metrics.jsonl` 中的 `seconds`、`netcdf_train_months`、`netcdf_validation_months` 和 `cache_gib`。缓存放得下时，第二轮两个 NetCDF 月份计数都应为 0。要比较新旧整轮性能，使用相同月份数、模型参数和硬件，分别记录第一轮与后续轮次。排查记录见 [train_01_log.md](../../runs/train_01_log.md)。

## 新版训练

```bash
bash /workspace/srtp-ocean/code/parallel_training/run_gpu_v2_6w.sh
```

使用完整 v2 的 80 个训练月和 24 个验证月，从头训练，最多 50 轮，连续 8 轮未改善则早停。启用 6 个训练和 2 个验证数据加载进程、1 个主进程 CPU 计算线程、CUDA 混合精度及 8 GiB 显存缓存。

使用 CPU 时，通过 `run_parallel.sh` 指定 `--device cpu --no-amp`。新划分和留出评估见 [数据集 v2](../dataset_v2/README.md)。

脚本在训练结束后用 `best.pt` 分别测试 2024 年和 2025 年上半年，生成 `test_metrics.json`。

## 输入消融（`ABLATE_MODE`，2026-11-21）

本目录新增了**可训练、可验证的输入消融机制**，用于评估各输入通道组（海表五源、内部 C 组上下文、位置偏移）对精度的贡献。

### 为什么不能用主进程 monkey-patch

`train_parallel.py` 的 DataLoader 通过 `multiprocessing_context="spawn"` 启动 worker。worker 是**新解释器、重新导入模块**，主进程对 `model_input` 的替换不会传进去，训练/验证仍读完整 276 通道（曾导致 `offsets_only` 与基线几乎一致）。

### 实现方式

`ablate_site/sitecustomize.py` 会在**每个 Python 解释器（含 spawn worker）启动时**自动运行：读取环境变量 `ABLATE_MODE`，包装 `server_training_data.TrainingDataset.load`，把对应通道组的**值与掩膜同时置零**。因此训练与验证用的都是消融后的输入，模型结构、通道数（276）与参数量（288,024）保持不变。

支持的 `ABLATE_MODE`：

| 模式 | 含义 |
| --- | --- |
| `full` | 不消融（默认） |
| `no_smap` / `no_wind` / `no_adt` | 去掉对应单个海表源 |
| `oisst_only` / `adt_only` / `wind_only` / `smap_only` | 仅保留一个海表源 |
| `no_surface` | 去掉全部海表源（保留内部上下文与偏移） |
| `no_context` | 去掉内部 C 组上下文（值+掩膜；仍保留位置偏移） |
| `offsets_only` | 仅保留 6 个位置偏移通道（无观测信息对照） |

### 单次运行

```bash
cd /g3240101584lt/srtp-ocean/code
export PYTHONPATH=$PWD/parallel_training/ablate_site:$PWD/python_pkgs:$PWD:$PWD/parallel_training
ABLATE_MODE=no_context /opt/conda/bin/python -u parallel_training/train_parallel.py \
  --dataset-dir /g3240101584lt/srtp-ocean/data/training_ready_v2 \
  --run-dir /g3240101584lt/srtp-ocean/runs/<新目录> \
  --device cuda --workers 1 --validation-workers 1 --cpu-threads 1 \
  --amp --cache-gib 8 --epochs 200 --patience 201 --seed 20261120
```

注意：
- `PYTHONPATH` **必须**包含 `parallel_training/ablate_site`（放在最前），否则 `sitecustomize` 不会加载；
- `ABLATE_MODE` 作为环境变量会在 spawn 子进程中继承；不要用 `--ablate` 之类的命令行参数代替；
- `ABLATE_MODE` 未设置或为 `full` 时行为与原始训练完全一致。

### 整批矩阵

```bash
bash parallel_training/run_input_matrix_v2.sh
```

默认跑 10 个输入变体（`MAX_PARALLEL=5`、`workers 1/1`、200 轮、`seed=20261120`），结果写入 `runs/ablation_input_v2_<UTC>/<mode>/`，每个变体带 `metrics.jsonl`、`best.pt`、`latest.pt`、日志和 `exitcode`。

### 结果核对与物理诊断

- 逐模式确认置零是否正确的恒等式：`full − no_adt = adt_only`、`full − no_wind = wind_only`、`full − no_smap = smap_only`，且四源之和等于 `full`；
- 物理约束一致性（违规率 vs CORA 参考场）用 `dataset_v2/physics_diagnostics.py`；
- 多套件结果汇总用 `dataset_v2/collect_ablation.py`。

### 已归档的失效实现

`_archive_broken_ablation_20261121/`（主进程 monkey-patch 版本，spawn 下失效，勿再使用）：
`ablate_input.py`、`run_ablation_suite.sh`、`run_input_matrix.sh`。
稀疏锚点原型另见 `_archive_sparse_anchor_20261121/`。维护时新增消融模式只需修改 `ablate_site/sitecustomize.py` 的 `_apply`，并在 `run_input_matrix_v2.sh` 的 `MODES` 中登记。
