# 训练模型代码索引

2026-11-23 更新。本文件持续记录 `code/` 下各模型的实现、训练入口、默认参数和共用依赖。新增或修改模型代码时同步更新本表；每次实际训练的轮次和结果记录在 [V2_Trainlog.md](../runs/V2_Trainlog.md) 和 [V4_Trainlog.md](../runs/V4_Trainlog.md)。精选配置、逐轮指标和测试JSON归档在 [`research/experiments/2026-11-23/source_artifacts/`](../research/experiments/2026-11-23/source_artifacts/)，模型权重和大日志不入库。

服务器代码目录：`/workspace/srtp-ocean/code/`。下文路径均相对此目录。

## 本次核对与协议入口（2026-11-22）

训练代码未在本次修改或运行。原主协议v2与新增v4分别保留：v2四模型入口为各模型目录的 `run_gpu.sh` 或 `parallel_training/run_four_models_200ep.sh`；v4入口为 `parallel_training/run_v4_four_models.sh`，通过 `train_arch.py` 复用统一训练循环。

v4启动脚本实际配置：200轮、seed20261120、patience201、2个训练worker／1个验证worker、14GiB缓存。它与旧v2脚本的6个训练worker不同，不应混写。训练和半年评估都需要将 `parallel_training/ablate_site` 放在 `PYTHONPATH` 前部，并设置 `ABLATE_MODE=no_smap`；单独指定v4数据目录不会屏蔽SMAP。

当前v4 `config.json` 未记录 `--arch` 选择或 `ABLATE_MODE`，半年报告也未记录清单／权重哈希；追溯需结合模型子目录、批量启动脚本和 `halfyear.log`，不能把环境约定当作配置文件已有字段。本次未补写实验配置或重跑验证。

输入消融的10个修正版训练均已完成；失效实现保存在 `_archive_broken_ablation_20261121/`，稀疏锚点原型代码保存在 `_archive_sparse_anchor_20261121/`。结果与文件位置见 [Runs_README.md](../runs/Runs_README.md)。

## 模型清单

| 模型 | 模型实现 | 训练／拟合代码 | 方法与默认规模 |
| --- | --- | --- | --- |
| Swin Transformer | `train_swin_physics.py` 中的 `SwinReconstructor` | `parallel_training/train_parallel.py` | 二维窗口注意力；dim=48、4个block、4个head、window=8 |
| 月气候态 | `climatology_baseline/model.py` | `climatology_baseline/train.py` | 复用训练期逐月背景，无迭代训练 |
| EOF＋岭回归 | `eof_ridge/model.py` | `eof_ridge/train.py` | 联合CT／SA垂向EOF，默认16模态、ridge alpha=0.01；一次闭式拟合 |
| 残差 CNN | `residual_cnn/model.py` | `residual_cnn/train.py` | 48通道、4个残差block；膨胀卷积 |
| U-Net | `unet_baseline/model.py` | `unet_baseline/train.py` | 起始24通道、3次下采样、跳跃连接；连续数值回归 |
| ViT | `vit_baseline/model.py` | `vit_baseline/train.py` | 全局注意力；3×3 patch、dim=96、4层、4个head，支持设定种子 |

月气候态忽略当前海表及Argo输入。EOF使用同位置的276个输入通道，拟合仅选训练期双变量60层都有效的完整列；不完整列和浅水列不参与拟合。EOF属于垂向模式分解，不是全域空间矩阵分解。

CNN、U-Net、ViT与当前并行Swin共用CORA网格监督、Argo Q真实位置监督和垂向稳定性损失。月气候态、EOF作为统计基线，拟合目标不同。

## 数据与模型接口

- 原主协议v2入口：`data/training_ready_v2/training_manifest.json`。训练80个月，验证24个月，2024年主测试12个月，2025年上半年补充测试6个月。
- 神经网络输入：当月及前两个月的海表变量、当月Argo C温盐网格、偏移与掩膜，形状 `[B, 276, 81, 321]`。
- 输出：标准化温盐异常，形状 `[B, 2, 60, 81, 321]`，顺序为CT、SA；反归一化并加回背景得到实际温盐。
- Q监督和E评分使用实际CORA经纬度轴；验证E用于选优，两个测试集在权重确定后评分。

完整来源与月份划分见 [Data_README.md](../data/Data_README.md)。

## CUDA启动入口

以下为CUDA运行入口；实际配置与成绩见训练简报。

| 模型 | 单次运行脚本 | 默认训练预算 |
| --- | --- | --- |
| Swin，六进程入口 | `parallel_training/run_gpu_v2_6w.sh` | 最多200轮，patience=8 |
| Swin，任务入口 | `parallel_training/run_task_500ep.sh` | 最多200轮，patience=8，可由环境变量覆盖；脚本文件名保留历史500ep标记 |
| 月气候态 | `climatology_baseline/run_gpu.sh` | 无迭代，记录epoch=0 |
| EOF＋岭回归 | `eof_ridge/run_gpu.sh` | 一次拟合，记录fit_passes=1 |
| 残差 CNN | `residual_cnn/run_gpu.sh` | 最多200轮，patience=500 |
| U-Net | `unet_baseline/run_gpu.sh` | 最多200轮，patience=500 |
| ViT | `vit_baseline/run_gpu.sh` | 最多200轮，patience=500 |

已有500轮Swin实验实际设置为 `epochs=500、patience=500`；它与目前默认200轮的启动脚本不同，复现实验以该运行目录的 `config.json` 为准。

新增五套入口均拒绝CPU回退；CNN／U-Net／ViT默认6个训练数据加载进程、2个验证加载进程、1个主CPU线程、AdamW学习率1e-4、CUDA AMP及8 GiB设备缓存。全部使用完整v2训练／验证月份。五套入口成功后，自动用最佳权重测试2024年及2025年上半年，并分别报告全部E和验证期未见E浮标。

## 种子与重复实验

新增五套的单次默认种子为 `20261117`。各目录 `run_seeds.sh` 默认顺序运行 `20261117、20261118、20261119`，汇总测试指标均值和样本标准差。

```bash
bash /workspace/srtp-ocean/code/residual_cnn/run_seeds.sh
bash /workspace/srtp-ocean/code/unet_baseline/run_seeds.sh
bash /workspace/srtp-ocean/code/vit_baseline/run_seeds.sh
```

月气候态和闭式EOF无随机初始化，通常一次即可，重复种子只用于复现核对。Python、NumPy和PyTorch已设置种子；CUDA采样反向传播仍可能非确定，不保证逐位一致。Swin入口支持单次SEED／--seed设置；本次未新增其多种子批量脚本。

## 共用代码与环境

| 位置 | 作用 |
| --- | --- |
| `server_training_data.py` | 按清单读取月场、Argo及掩膜 |
| `parallel_training/train_parallel.py` | 既有并行加载、缓存、损失及验证函数 |
| `dataset_v2/grid_sampler.py` | 非均匀原生经纬度采样 |
| `dataset_v2/evaluate_checkpoint.py` | Swin评估及共用指标累计 |
| `climatology_baseline/runtime.py` | 新增模型共用CUDA检查、种子和神经网络训练循环 |
| `climatology_baseline/evaluate.py` | 新增四模型的统一评估入口 |
| `climatology_baseline/summarize_seeds.py` | 汇总已完成的多种子JSON报告 |
| `climatology_baseline/launch_model.sh`、`launch_seeds.sh` | 四模型的单次及顺序重复运行调度 |

启动脚本使用 `/opt/conda/bin/python`，设置 `PYTHONPATH` 加载现有 `python_pkgs/`、代码根目录及 `dataset_v2/`。五个新增目录需与既有共用代码一起保留。依赖归档存于 `dependency_archives/`；数据上传归档位于 `data/zip_s/`，训练继续读取解包数据。

历史入口 `run_swin_training.sh`／`train_swin_physics.py` 保留v1复现用途，其旧纬度采样与当前v2入口不同。数据准备脚本为 `prepare_server_training_data.py`、`run_server_preprocessing.sh`，样例查看工具为 `inspect_server_data.py`。

## 检查与文档维护

新增模型代码已完成Ruff、严格mypy、Python编译语法和shell语法检查，2026-11-20已完成本次指定训练和测试，详见下方更新。详见 [静态检查](climatology_baseline/STATIC_CHECKS.md) 及各模型目录的README。

后续维护本文件时，更新模型清单、训练入口、默认超参数、种子及检查状态。具体运行配置、权重和成绩保存在 `runs/`，见 [Runs_README.md](../runs/Runs_README.md) 和 [V2_Trainlog.md](../runs/V2_Trainlog.md) 和 [V4_Trainlog.md](../runs/V4_Trainlog.md)。

2026-11-19：新增ViT，支持 `--seed`、`SEED` 和三种子顺序运行。神经网络Python默认及对应启动脚本默认轮次统一为250；月气候态／EOF不做迭代。模型实跑记录按日期见下方更新，既有实验配置保留。

<!-- BASELINE_SUITE_20261120 -->
2026-11-20：五模型已完成当前指定运行及两段测试；CNN2种子、U-Net／ViT各1种子，神经网络均250轮。后续种子已取消，无新训练。批量调度入口为 `climatology_baseline/run_all_models.sh`；成绩见训练简报。
<!-- END_BASELINE_SUITE_20261120 -->
## Swin 随机种子

2026-11-20：原有 `--seed` 默认20261117，现已补齐Python、NumPy、PyTorch CPU／CUDA和DataLoader worker种子。训练和验证使用独立的显式generator；启用尽力确定性设置，CUDA grid_sample反向仍可能非确定。两个CUDA脚本支持 `SEED=42 bash /workspace/srtp-ocean/code/parallel_training/run_gpu_v2_6w.sh`；任务入口 `run_task_500ep.sh` 同样支持SEED；直接Python入口使用 `--seed 42`。默认最多200轮，随机种子写入config.json。

代码通过Ruff、严格mypy（两处PyTorch 2.7注解缺失作定点标注）、Python／shell语法及隔离CPU随机数重复性检查；未执行模型或CUDA负载。既有损失、验证和缓存逻辑未修改，其他模型原加载函数调用保持兼容。检查配置位于 `parallel_training/pyproject.toml`。

2026-11-20：残差CNN、U-Net、ViT、Swin后续训练默认上限统一为200轮，保留原提前停止设置；历史配置、成绩和v2数据划分不变，本次未启动训练。

2026-11-20：CNN、U-Net、ViT、Swin新种子20261120各200轮及两段测试已完成；旧成果69文件（含28权重）校验值不变。结果目录 runs/four_models_200ep_seed20261120_20261120T115114Z/，未安排后续训练。

本轮单次批量调度：code/parallel_training/run_four_models_200ep.sh（各模型仅跑新种子20261120一次）。

2026-11-20：训练简报已按最佳实验、四模型分类及时间顺序重排，收录19条记录；仅整理文档。

## 空间／分深度诊断与温盐切片

2026-11-20：新增只读诊断与切片包 `spatial_depth_diagnostics/`（入口 `spatial_depth_diagnostics/run_diagnostics.sh`，说明见 [README.md](spatial_depth_diagnostics/README.md)），兼容全部 6 套 v2 权重及直接 CORA／背景场，支持 `--device auto|cpu|cuda`：
- `diagnose`：输出 5–500 m 分深度误差曲线（CT／SA 的 RMSE、Bias、R²）、留出 Argo E 空间误差分布图及代表性剖面重建对比图，并导出 `depth_metrics.csv`、`spatial_error_bins.csv`、`representative_profiles.json`。
- `slice`：输出海洋 CT／SA 水平深度切片、纬向垂向断面（固定纬度）与经向垂向断面（固定经度），可选附带 CORA 差值及气候态异常图，并支持导出 NetCDF4（`.nc`）与 `.npz` 切片数据。
- 已完成 Ruff 风格／格式检查、`bash -n`、`py_compile`、6 项合成数据单元测试（含 NetCDF4／NPZ 读写与全部图表渲染检查）及 6 套真实模型权重只读加载核验；未启动模型训练或全量测试评估。

## 输入源与物理约束消融（2026-11-21）

### 背景：一次失效实现与修复

`parallel_training/train_parallel.py` 的 DataLoader 使用 `multiprocessing_context="spawn"`。此前用“在主进程 monkey-patch `model_input`”的方式做输入消融，patch 不会进入 worker 进程，训练/验证实际读到的仍是完整 276 通道；表现是 `offsets_only`（仅位置偏移）与基线几乎一致。该批结果已作废。

现改为 **`sitecustomize` 注入**：`parallel_training/ablate_site/sitecustomize.py` 在每个 Python 解释器（含 spawn worker）启动时读取环境变量 `ABLATE_MODE`，包装 `server_training_data.TrainingDataset.load`，把指定通道组的值与掩膜同时置零。已用逐模式求和验证：`full − no_adt = adt_only`、`full − no_wind = wind_only`、`full − no_smap = smap_only` 且四源之和等于 full，通道索引无错位。

### 新增文件

| 文件 | 作用 |
| --- | --- |
| `parallel_training/ablate_site/sitecustomize.py` | 消融注入点。支持 `ABLATE_MODE = full / no_smap / no_wind / no_adt / no_context / no_surface / offsets_only / oisst_only / adt_only / wind_only / smap_only`；训练与验证进程同时生效。 |
| `parallel_training/run_input_matrix_v2.sh` | 输入消融矩阵启动器：10 个变体、`MAX_PARALLEL=5`、`workers 1/1`、200 轮、`seed=20261120`，结果写入 `runs/ablation_input_v2_<UTC>/<mode>/`。 |
| `dataset_v2/physics_diagnostics.py` | 物理一致性诊断：加载 checkpoint，在指定 split 上计算垂向稳定 proxy 的违规率、违规幅度，并给出 CORA 参考场违规率作对照，输出 JSON。 |
| `dataset_v2/collect_ablation.py` | 汇总多套件（λ 扫描、输入消融、基线）的 best 指标，输出 `summary.json` 供报告生成脚本使用。 |
| `dataset_v2/halfyear_eval.py` | 按半年评估 E 组剖面（CT/SA RMSE、bias、R²、归一化分数）。 |
| `dataset_v2/halfyear_eval2.py` | 逐月 + 半年评估，并统计每月观测异常幅度（用于区分“时段变难”与“模型退化”）。 |
| `dataset_v2/input_coverage.py` | 逐月统计 5 个海表源的掩膜覆盖率，排查输入数据缺口。 |
| `dataset_v2/diagnose_month.py` | 指定月份的逐剖面 / 逐深度误差诊断，导出 CSV 与深度 JSON。 |
| `dataset_v2/depth_band_eval.py` | 42 个月 × 5 个模型 × 5 个深度带的 RMSE / bias 评估。 |

### 已移除的失效实现

以下文件因依赖主进程 monkey-patch、在 spawn worker 下失效，已从代码路径移入 `parallel_training/_archive_broken_ablation_20261121/`（保留可查，不再使用）：

- `parallel_training/ablate_input.py`
- `parallel_training/run_ablation_suite.sh`
- `parallel_training/run_input_matrix.sh`

稀疏锚点方向的 4 个原型文件（`sparse_anchor_model.py`、`train_sparse_anchor.py`、`train_sparse_anchor_gpu.py`、`evaluate_sparse_anchor.py`）已于同日移入 `parallel_training/_archive_sparse_anchor_20261121/`，对应 200 轮 GPU 结果仍保留在 `runs/sparse_anchor_200ep_gpu_20261121T055809Z/`。

### 使用方法

输入消融（任一变体，独立运行）：

```bash
cd /g3240101584lt/srtp-ocean/code
export PYTHONPATH=$PWD/parallel_training/ablate_site:$PWD/python_pkgs:$PWD:$PWD/parallel_training
ABLATE_MODE=no_context /opt/conda/bin/python -u parallel_training/train_parallel.py \
  --dataset-dir /g3240101584lt/srtp-ocean/data/training_ready_v2 \
  --run-dir /g3240101584lt/srtp-ocean/runs/<新目录> \
  --device cuda --workers 1 --validation-workers 1 --cpu-threads 1 \
  --amp --cache-gib 8 --epochs 200 --patience 201 --seed 20261120
```

整批矩阵：`bash parallel_training/run_input_matrix_v2.sh`（可用 `MAX_PARALLEL` 覆盖并发数）。

物理诊断：

```bash
python dataset_v2/physics_diagnostics.py --checkpoint <run>/best.pt \
  --output <run>.physics.json --device cpu --cpu-threads 8
```

结果汇总：`python dataset_v2/collect_ablation.py > summary.json`。

### 结果与产物

- 有效输入消融批次：`runs/ablation_input_v2_20261121T074157Z/`（10 个变体，全部 `exit=0`）。
- λ 扫描（`runs/ablation_20261121T061720Z/A1_lambda*`）：λ=0 / 0.1 / 1.0，配置一致。
- 作废批次（仅作记录）：`runs/ablation_20261121T061720Z` 中的输入消融变体、`runs/ablation_surface_*/`。
- 结论摘要（单种子，验证集）：内部 C 组上下文最重要（去掉 +6.78%），去掉全部海面仅 +1.02%；单源保留 ≤ +1.03%（只留 ADT 仅 +0.08%）；无信息对照（仅位置偏移）+24.20%；物理 λ 从 0→1.0 违规率下降约 12%，验证分数仅差 0.06%。

后续维护：新增消融变体时，只需在 `sitecustomize.py` 的 `_apply` 中增加模式，并在启动器 `MODES` 列表中加入对应名称；不要再用主进程 monkey-patch 的方式。

## v4 数据集与四模型统一入口（2026-11-21）

本节补充 v4 扩展协议（训练 2005–2019）相关代码。输入消融机制（`ablate_site/sitecustomize.py`、`ABLATE_MODE`）见上一节，本节不重复。

### 新增文件

| 文件 | 作用 |
| --- | --- |
| `dataset_v2/build_dataset_v4.py` | 构建 `training_ready_v4`：方案 A 划分（训练 179 月 / 验证 24 / 测试 36 / 前向 6），`fit` 重拟合背景与归一化、`publish` 生成清单与审计；只读 v1 的 CORA/Argo 月文件，不覆盖 v2 |
| `parallel_training/train_arch.py` | 四模型统一训练入口：复用 `train_parallel.train()`，仅把模型工厂换成 `--arch swin|cnn|unet|vit`；其余参数与 Swin 入口一致（含 `--seed`、`--dataset-dir`、`--cache-gib` 等） |
| `dataset_v2/halfyear_eval_v4.py` | 半年节点评估（2020H1–2025H1）：读取 v4、加载各模型 `best.pt`，按 E 组剖面输出每半年 CT/SA RMSE、bias、R²、归一化分数 |
| `parallel_training/run_v4_four_models.sh` | v4 四模型批量入口：4 路并行、200 轮、seed=20261120、`ABLATE_MODE=no_smap`、`cache-gib=14`，结果写入 `runs/v4A_four_models_<UTC>/` |
| `parallel_training/run_build_v4.sh` | v4 数据集构建入口（`fit` → `publish`，日志写入 `data/build_v4.log`） |
| `parallel_training/run_publish_v4.sh` | 仅执行 v4 的 `publish` 阶段（`fit` 已完成时使用） |
| `parallel_training/check_v4_progress.sh` | 打印 v4 四模型进度（`模型:已完成轮数 [exit=N]`），供训练监控使用 |

### 用法

```bash
# 1) 构建 v4（CPU，约 2 分钟）
bash code/parallel_training/run_build_v4.sh

# 2) 四模型训练（GPU，4 路并行，约 25–60 分钟）
bash code/parallel_training/run_v4_four_models.sh

# 3) 半年评估（CPU，约 20–60 秒；必须带 ABLATE_MODE=no_smap 与 ablate_site）
cd code
PYTHONPATH=$PWD/parallel_training/ablate_site:$PWD/python_pkgs:$PWD:$PWD/parallel_training \
ABLATE_MODE=no_smap python dataset_v2/halfyear_eval_v4.py \
  --suite /workspace/srtp-ocean/runs/v4A_four_models_20261121T093236Z \
  --output /workspace/srtp-ocean/runs/v4A_four_models_20261121T093236Z/halfyear.json
```

注意：

- `--arch` 之外的参数与 `train_parallel.py` 完全一致；CNN 用 `--dim 48 --blocks 4`，U-Net 用 `--dim 24`，ViT 用 `--dim 96 --blocks 4 --heads 4`，Swin 用 `--dim 48 --blocks 4 --heads 4 --window 8`；
- v4 全流程无 SMAP（`ABLATE_MODE=no_smap`），评估时必须同样设置，否则输入规格与训练不一致；
- v4 与 v2 的验证期不同（2020–2021 vs 2022–2023），两者分数不可直接比较。

### 结果（2026-11-21）

`runs/v4A_four_models_20261121T093236Z/`：四模型各 200 轮、`exit=0`；最佳验证分数 CNN 1.4680、Swin 1.4719、U-Net 1.5215、ViT 1.5610；测试/前向 CT RMSE 见 [Runs_README](../runs/Runs_README.md)。

`code/parallel_training/README.md` 同时追加了输入消融（`ABLATE_MODE`）的使用说明。

<!-- V4_SEED_20261122 -->
## v4 新种子训练（2026-11-22，已完成）

本次入口为 `parallel_training/run_v4_four_models_seed20261122.sh`：v4、seed20261122、no_smap、四模型各200轮，2个训练worker／1个验证worker、14GiB缓存、CUDA AMP。四模型已全部正常完成。模型实现和训练源码哈希与启动前一致；脚本固定本次运行目录，拒绝覆盖已有suite.log，不用于再次启动同名实验。
<!-- END_V4_SEED_20261122 -->

## v2 seed=20261118 三模型并行训练（2026-11-23，已完成）

新增并行入口 `parallel_training/run_v2_three_models_seed20261118.sh`（SHA-256 `9195cc97b7774d6d52b9e9143d8534559e3f9381a8fe7ea3fd41919913ea997a`）：v2 协议、`ABLATE_MODE` 未启用、`SEED=20261118`、`EPOCHS=200`、`PATIENCE=201`、6 训练 worker／2 验证 worker、8 GiB 缓存。并列调度 `swin`（`run_task_500ep.sh`）、`unet_baseline` 与 `vit_baseline`（各自 `run_gpu.sh`），并在训练结束后对各自 `best.pt` 自动执行 `test_locked` 与 `forward_test` 评估。三模型均已完成且退出码为 `0`，输出目录为 `runs/three_models_v2_seed20261118_20261123T054818Z/`，原有代码与历史权重未修改。

## v4 Swin／CNN Argo C Context Dropout（2026-11-23，已就绪待跑）

为在不改动现有 `train_parallel.py`、`train_arch.py` 与历史模型实现的前提下支持 `v4` 训练期 Argo C 随机丢弃增强，新增以下文件：

| 文件 | 作用 |
| --- | --- |
| `parallel_training/context_dropout.py` | 动态 Argo C Context Dropout 核心实现（`mixed`／`pointwise`／`block`／`full_drop`／`none`），只读克隆不篡改 `MonthCache`，丢弃格点同步置零 `30:276`（含值、掩膜与位置／时间偏移掩膜），保留 `0:30` 海表通道；内置 `assert_no_smap_input` 强制校验 `v4` 无 SMAP |
| `parallel_training/train_v4_context_dropout.py` | `v4` Swin／残差 CNN 专用训练与评估入口：复用现有模型与损失函数，强制 `ABLATE_MODE=no_smap`，将 `arch`、`ablate_mode` 与 `context_dropout` 参数写入 `config.json`；训练结束后自动对 `best.pt` 执行 `test_locked` 与 `forward_test` 评估（含全部 E 与验证期未见 E，以及可选 `test_metrics_no_context.json` 纯海表反演测试） |
| `parallel_training/tests/test_context_dropout.py` | 6 项 `unittest` 单元测试（覆盖通道边界、不篡改缓存、偏移掩膜联动置零、确定性与 Swin／CNN 前向反向检查） |
| `parallel_training/run_v4_swin_cnn_context_dropout.sh` | `v4` Swin＋CNN 并行启动脚本（默认 `SEED=20261123`、`EPOCHS=200`、`PATIENCE=201`、`mixed` 模式、`ABLATE_MODE=no_smap`） |

已完成 `bash -n`、`py_compile`、6 项单元测试（1.736s 全部通过）与 `training_ready_v4`（`201801`）真实单月 `no_smap` 联调；尚未启动正式训练。
