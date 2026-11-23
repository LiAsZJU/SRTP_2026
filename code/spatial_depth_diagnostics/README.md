# 空间／分深度诊断与海洋温盐切片模块（`spatial_depth_diagnostics`）

独立只读分析与可视化工具，用于补充汇总型 `training_visualization/` 的细粒度诊断与海洋三维场二维切片输出。

- **只读不改源**：不修改任何权重、日志或数据目录；每次输出必须指定全新的 `--output-dir`（已存在时拒绝覆盖）。
- **全模型兼容**：统一支持 v2 全部 6 套模型权重（`Swin`、`残差CNN`、`U-Net`、`ViT`、`月气候态`、`EOF＋岭回归`），并校验 `training_manifest.json` 的 SHA-256。
- **CPU / CUDA 通用**：通过 `--device auto|cpu|cuda` 控制设备；从不启动训练。
- **严格沿用 v2 口径**：使用原生非均匀经纬度双线性采样 `dataset_v2/grid_sampler.py`，并区分 `all_E` 与 `E_platforms_unseen_in_validation`。留出 Argo E 与 CORA 上游观测存在重合，指标仍按留出 Argo 口径解释。

---

## 一、功能清单

### 1. `diagnose`：分深度误差 + 空间误差分布 + 典型剖面重建对比
- **分深度误差曲线（5–500 m）**：按 60 个原生深度层分别统计 CT / SA 的 RMSE、Bias、R² 与有效点数，支持多模型叠加绘制；输出 `depth_metrics.csv` 与 `figures/depth_error_<split>_<cohort>.{png,pdf}`。
- **空间误差分布图**：将留出 Argo E 误差按指定经纬度网格（默认 `2.0° × 2.0°`）与 4 个深度层段（`full_5_500m`、`surface_5_50m`、`thermocline_55_200m`、`subsurface_210_500m`）统计并绘制 2×2 空间 RMSE / Bias 分布图；空网格保留 `NaN` 不填零；输出 `spatial_error_bins.csv` 与 `figures/spatial_error_*.{png,pdf}`。
- **典型剖面重建对比图**：按参考模型误差分位数分层选取代表性 Argo E 剖面（由低误差到高误差，同时叠加气候态背景与全部对比模型），输出 `representative_profiles.json` 与 `figures/representative_profiles_<split>_<cohort>.{png,pdf}`。

### 2. `slice`：海洋温度（CT）／盐度（SA）二维切片输出（图 + 数据）
支持对**模型重建场**（`--model label=checkpoint.pt`）或**直接海洋数据场**（`--direct-source cora background`）提取并导出三种切片：
- **水平深度切片**（`--depths`）：指定深度（5–500 m，如 `5 50 100 200 500`）的 `latitude × longitude` 平面分布图。
- **纬向垂向剖面（沿固定纬度）**（`--zonal-lats`）：如赤道 `0.0`、`±5.0°N` 的 `depth × longitude` 东西向温盐跃层断面图（叠加等温／等盐线）。
- **经向垂向剖面（沿固定经度）**（`--meridional-lons`）：如 `165.0`、`190.0`（170°W）、`220.0`（140°W）、`250.0`（110°W，也支持负西经 `-110` 自动换算为 `250°E`）的 `depth × latitude` 跨赤道断面图。
- **可选对比与异常**：`--include-cora` 自动附带 CORA 参考与 `Source - CORA` 差值场；`--include-anomaly` 自动附带相对训练期月气候态的异常场 `Source - Clim`。
- **结构化数据导出**：通过 `--export-data nc npz` 同时导出带完整坐标、单位、缺失掩膜与匹配坐标属性的 **NetCDF4（`.nc`）** 和 **压缩 NumPy（`.npz`）** 文件（位于 `slices/`），可直接用 Python / Panoply / QGIS / ArcGIS 读取。

---

## 二、使用示例（服务器）

### 示例 1：2024 主测试集多模型分深度／空间／剖面诊断
```bash
bash /workspace/srtp-ocean/code/spatial_depth_diagnostics/run_diagnostics.sh diagnose \
  --dataset-dir /workspace/srtp-ocean/data/training_ready_v2 \
  --model swin=/workspace/srtp-ocean/runs/four_models_200ep_seed20261120_20261120T115114Z/swin/best.pt \
  --model cnn=/workspace/srtp-ocean/runs/four_models_200ep_seed20261120_20261120T115114Z/residual_cnn/best.pt \
  --model unet=/workspace/srtp-ocean/runs/four_models_200ep_seed20261120_20261120T115114Z/unet_baseline/best.pt \
  --model vit=/workspace/srtp-ocean/runs/four_models_200ep_seed20261120_20261120T115114Z/vit_baseline/best.pt \
  --splits test_locked \
  --device auto \
  --output-dir /workspace/srtp-ocean/runs/diagnostics/four_models_200ep_2024
```
如在纯 CPU 上先快速抽查某两个月，可把 `--splits test_locked` 改为 `--months 202401 202407`。

### 示例 2：输出 Swin 重建场的水平深度切片 + 赤道／经向垂向断面（含 CORA 差值、气候态异常与 NetCDF/NPZ 数据）
```bash
bash /workspace/srtp-ocean/code/spatial_depth_diagnostics/run_diagnostics.sh slice \
  --dataset-dir /workspace/srtp-ocean/data/training_ready_v2 \
  --model swin=/workspace/srtp-ocean/runs/four_models_200ep_seed20261120_20261120T115114Z/swin/best.pt \
  --months 202401 202407 \
  --depths 5 50 100 200 500 \
  --zonal-lats 0.0 5.0 -5.0 \
  --meridional-lons 165.0 190.0 220.0 250.0 \
  --include-cora --include-anomaly \
  --export-data nc npz \
  --device auto \
  --output-dir /workspace/srtp-ocean/runs/diagnostics/swin_slices_2024
```

### 示例 3：直接对原始 CORA 或训练期气候态背景切出温盐切片（无需加载模型权重）
```bash
bash /workspace/srtp-ocean/code/spatial_depth_diagnostics/run_diagnostics.sh slice \
  --dataset-dir /workspace/srtp-ocean/data/training_ready_v2 \
  --direct-source cora \
  --months 202401 \
  --depths 5 100 300 \
  --zonal-lats 0.0 \
  --include-anomaly \
  --export-data nc npz \
  --output-dir /workspace/srtp-ocean/runs/diagnostics/cora_slices_202401
```

---

## 三、自检与测试

```bash
PYTHONPATH=/workspace/srtp-ocean/code:/workspace/srtp-ocean/code/dataset_v2:/workspace/srtp-ocean/code/python_pkgs \
  /opt/conda/bin/python -m unittest discover -s /workspace/srtp-ocean/code/spatial_depth_diagnostics/tests -v
```
