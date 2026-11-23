#!/usr/bin/env python3
"""逐月 + 半年评估：模型误差 vs 观测异常幅度（判断"时段变难"还是"模型退化"）。

不修改现有文件。输出 JSON：
  periods: 半年序列
  models[model][period]: 半年池化指标
  monthly[month]: {support, anomaly_rms, models{model: metrics}, climatology: metrics}

用法（服务器，CPU）：
  cd /g3240101584lt/srtp-ocean/code
  export PYTHONPATH=$PWD/python_pkgs:$PWD
  /opt/conda/bin/python -u dataset_v2/halfyear_eval2.py --output /tmp/hy2.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset_v2.grid_sampler import GridSampler  # noqa: E402
from residual_cnn.model import ResidualCNN  # noqa: E402
from server_training_data import TrainingDataset  # noqa: E402
from train_swin_physics import Objective, SwinReconstructor, as_tensor, model_input  # noqa: E402
from unet_baseline.model import UNetReconstructor  # noqa: E402
from vit_baseline.model import ViTReconstructor  # noqa: E402

DATASET = Path("/g3240101584lt/srtp-ocean/data/training_ready_v2")
SUITE = Path("/g3240101584lt/srtp-ocean/runs/four_models_200ep_seed20261120_20261120T115114Z")
RUNS = {
    "swin500": Path("/g3240101584lt/srtp-ocean/runs/swin_physics_v2_500ep_gpu_20261118T044643Z"),
    "swin200": SUITE / "swin",
    "cnn200": SUITE / "residual_cnn",
    "unet200": SUITE / "unet_baseline",
    "vit200": SUITE / "vit_baseline",
}
KINDS = {"swin500": "swin", "swin200": "swin", "cnn200": "cnn", "unet200": "unet", "vit200": "vit"}
PERIODS = [
    ("2022H1", "202201", "202206"), ("2022H2", "202207", "202212"),
    ("2023H1", "202301", "202306"), ("2023H2", "202307", "202312"),
    ("2024H1", "202401", "202406"), ("2024H2", "202407", "202412"),
    ("2025H1", "202501", "202506"),
]


def build_model(kind: str, config: dict) -> torch.nn.Module:
    if kind == "swin":
        return SwinReconstructor(276, config["dim"], config["blocks"], config["heads"], config["window"])
    if kind == "cnn":
        return ResidualCNN(config["width"], config["blocks"])
    if kind == "unet":
        return UNetReconstructor(config["width"])
    return ViTReconstructor(config["embedding_dim"], config["blocks"], config["heads"], config.get("dropout", 0.0))


class Accum:
    """按变量累计：count, sum(y), sum(y^2), sum(e), sum(e^2), sum(e_norm^2)。"""

    def __init__(self) -> None:
        self.v = np.zeros((2, 6), dtype="f8")
        self.profiles = 0
        self.platforms: set[str] = set()

    def add(self, obs, pred, norm_err, valid, role, platform) -> None:
        cohort = role == 2
        good = cohort & valid.any(axis=(1, 2))
        self.profiles += int(good.sum())
        self.platforms.update(platform[good].tolist())
        for i in range(2):
            mask = cohort[:, None] & valid[:, i]
            y = obs[:, i][mask].astype("f8")
            if y.size == 0:
                continue
            e = pred[:, i][mask].astype("f8") - y
            ne = norm_err[:, i][mask].astype("f8")
            self.v[i] += (y.size, y.sum(), (y * y).sum(), e.sum(), (e * e).sum(), (ne * ne).sum())

    def summary(self) -> dict:
        out = {"profiles": self.profiles, "platforms": len(self.platforms)}
        total = float(self.v[:, 0].sum())
        out["normalized_score"] = math.sqrt(self.v[:, 5].sum() / total) if total else None
        for i, name in enumerate(("CT", "SA")):
            count, total_y, sumsq, esum, sse, _ = self.v[i]
            if count:
                sst = sumsq - total_y * total_y / count
                out[name] = {
                    "points": int(count),
                    "rmse": math.sqrt(sse / count),
                    "bias": esum / count,
                    "r2": 1 - sse / sst if sst > 0 else None,
                }
            else:
                out[name] = None
        return out


def period_of(month: str) -> str:
    return f"{month[:4]}{'H1' if int(month[4:]) <= 6 else 'H2'}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cpu")

    source = TrainingDataset(args.dataset_dir)
    reference = next(iter(source.records.values()))
    sampler = GridSampler(Path(reference["background"]), device)
    objective = Objective(source, device, 1.0, 0.01, 1e-4, 0.01)
    mean = objective.mean[0, :, :, 0, 0]
    std = objective.std[0, :, :, 0, 0]

    models: dict[str, torch.nn.Module] = {}
    for name, run_dir in RUNS.items():
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        model = build_model(KINDS[name], ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[name] = model

    half = {name: {p[0]: Accum() for p in PERIODS} for name in list(RUNS) + ["climatology"]}
    monthly: dict[str, dict] = {}

    for _, start, end in PERIODS:
        y, mo = int(start[:4]), int(start[4:])
        while f"{y:04d}{mo:02d}" <= end:
            month = f"{y:04d}{mo:02d}"
            mo += 1
            if mo > 12:
                y, mo = y + 1, 1
            if month not in source.records:
                continue
            sample = source.load(month)
            role, platform = _load_role(source.records[month]["argo_profiles"])
            valid = sample["profile_mask"].astype(bool)
            obs = as_tensor(sample["profile_absolute"], device)
            positions = as_tensor(sample["profile_position"], device)
            background = as_tensor(sample["profile_background"], device)
            grid = model_input(sample, device)
            period = period_of(month)

            # 观测异常幅度（相对训练期气候态），衡量"这个月本身有多难"
            anomaly = (obs - background).cpu().numpy()
            anomaly_rms = {}
            for i, name in enumerate(("CT", "SA")):
                mask = (role == 2)[:, None] & valid[:, i]
                anomaly_rms[name] = float(np.sqrt(np.mean(anomaly[:, i][mask] ** 2))) if mask.any() else None

            month_acc = {name: Accum() for name in list(RUNS) + ["climatology"]}
            clim = background.cpu().numpy()
            _add(month_acc["climatology"], obs.cpu().numpy(), clim, np.zeros_like(clim), valid, role, platform)
            _add(half["climatology"][period], obs.cpu().numpy(), clim, np.zeros_like(clim), valid, role, platform)

            for name, model in models.items():
                with torch.no_grad():
                    prediction = model(grid)
                    normalized = sampler(prediction.float(), positions)
                    physical = (background + normalized * std + mean).cpu().numpy()
                    norm_err = normalized.cpu().numpy() - sample["profile_values"]
                _add(month_acc[name], obs.cpu().numpy(), physical, norm_err, valid, role, platform)
                _add(half[name][period], obs.cpu().numpy(), physical, norm_err, valid, role, platform)

            monthly[month] = {
                "period": period,
                "profiles": month_acc["swin200"].profiles,
                "platforms": len(month_acc["swin200"].platforms),
                "anomaly_rms": anomaly_rms,
                "models": {name: month_acc[name].summary() for name in RUNS},
                "climatology": month_acc["climatology"].summary(),
            }
            print(month, "ok", flush=True)

    payload = {
        "periods": [p[0] for p in PERIODS],
        "models": {name: {p: item.summary() for p, item in blocks.items()} for name, blocks in half.items()},
        "monthly": monthly,
    }
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", args.output, flush=True)


def _load_role(path: str):
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        return np.asarray(ds["role_code"][:]), np.asarray(ds["platform_number"][:], dtype=str)


def _add(accum: Accum, obs, pred, norm_err, valid, role, platform) -> None:
    accum.add(obs, pred, norm_err, valid, role, platform)


if __name__ == "__main__":
    main()
