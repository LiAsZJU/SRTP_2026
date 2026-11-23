#!/usr/bin/env python3
"""(2026-11-21) 按半年（6 个月）分块评估四模型 + 气候态基线。

不修改任何现有文件。输出：
  - JSON: 每个模型 × 每个半年块的 E 组剖面指标（RMSE / bias / R² / 归一化分数）
  - 同时输出逐月指标，便于画时间序列

用法（服务器，CPU 即可）：
  cd /g3240101584lt/srtp-ocean/code
  export PYTHONPATH=$PWD/python_pkgs:$PWD
  /opt/conda/bin/python -u dataset_v2/halfyear_eval.py --output /tmp/halfyear.json
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
PERIODS = [
    ("2022H1", "202201", "202206"),
    ("2022H2", "202207", "202212"),
    ("2023H1", "202301", "202306"),
    ("2023H2", "202307", "202312"),
    ("2024H1", "202401", "202406"),
    ("2024H2", "202407", "202412"),
    ("2025H1", "202501", "202506"),
]


def build_model(kind: str, config: dict) -> torch.nn.Module:
    if kind == "swin":
        return SwinReconstructor(276, config["dim"], config["blocks"], config["heads"], config["window"])
    if kind == "cnn":
        return ResidualCNN(config["width"], config["blocks"])
    if kind == "unet":
        return UNetReconstructor(config["width"])
    if kind == "vit":
        return ViTReconstructor(config["embedding_dim"], config["blocks"], config["heads"], config.get("dropout", 0.0))
    raise ValueError(kind)


class Accum:
    def __init__(self) -> None:
        self.values = np.zeros((2, 6), dtype="f8")
        self.profiles = 0
        self.platforms: set[str] = set()

    def add(self, obs, pred, norm_err, valid, role, platform) -> None:
        cohort = role == 2
        good = cohort & valid.any(axis=(1, 2))
        self.profiles += int(good.sum())
        self.platforms.update(platform[good].tolist())
        for v in range(2):
            mask = cohort[:, None] & valid[:, v]
            y = obs[:, v][mask].astype("f8")
            e = pred[:, v][mask].astype("f8") - y
            ne = norm_err[:, v][mask].astype("f8")
            self.values[v] += (y.size, y.sum(), (y * y).sum(), e.sum(), (e * e).sum(), (ne * ne).sum())

    def summary(self) -> dict:
        out = {"profiles": self.profiles, "platforms": len(self.platforms)}
        norm_count = float(self.values[:, 0].sum())
        out["normalized_score"] = math.sqrt(self.values[:, 5].sum() / norm_count) if norm_count else None
        for v, name in enumerate(("CT", "SA")):
            count, total, sumsq, esum, sse, _ = self.values[v]
            if count:
                sst = sumsq - total * total / count
                out[name] = {
                    "points": int(count),
                    "rmse": math.sqrt(sse / count),
                    "bias": esum / count,
                    "r2": 1 - sse / sst if sst > 0 else None,
                }
            else:
                out[name] = None
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, default=DATASET)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)

    source = TrainingDataset(args.dataset_dir)
    reference = next(iter(source.records.values()))
    sampler = GridSampler(Path(reference["background"]), device)
    objective = Objective(source, device, 1.0, 0.01, 1e-4, 0.01)
    mean = objective.mean[0, :, :, 0, 0]      # (2, 60)
    std = objective.std[0, :, :, 0, 0]        # (2, 60)

    models: dict[str, tuple[torch.nn.Module, str]] = {}
    for name, run_dir in RUNS.items():
        checkpoint = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        kind = {"swin500": "swin", "swin200": "swin", "cnn200": "cnn", "unet200": "unet", "vit200": "vit"}[name]
        model = build_model(kind, config).to(device)
        model.load_state_dict(checkpoint["model"])
        model.eval()
        models[name] = (model, kind)
        print(f"loaded {name} epoch={checkpoint.get('epoch')} params={sum(p.numel() for p in model.parameters())}", flush=True)

    acc = {name: {period[0]: Accum() for period in PERIODS} for name in list(RUNS) + ["climatology"]}
    monthly: dict[str, dict] = {}

    months: list[str] = []
    for _, start, end in PERIODS:
        cursor = start
        while cursor <= end:
            months.append(cursor)
            year, month = int(cursor[:4]), int(cursor[4:])
            cursor = f"{year + (month == 12):04d}{(month % 12) + 1:02d}"

    for month in months:
        if month not in source.records:
            continue
        sample = source.load(month)
        role, platform = _load_role(source.records[month]["argo_profiles"])
        valid = sample["profile_mask"].astype(bool)
        obs = as_tensor(sample["profile_absolute"], device)
        positions = as_tensor(sample["profile_position"], device)
        background = as_tensor(sample["profile_background"], device)
        grid = model_input(sample, device)
        period = _period_of(month)

        # 气候态基线：直接用背景场在剖面位置的值
        clim = background.cpu().numpy()
        _accumulate(acc["climatology"][period], obs.cpu().numpy(), clim, np.zeros_like(clim), valid, role, platform)

        record = {"month": month, "period": period}
        for name, (model, _) in models.items():
            with torch.no_grad():
                prediction = model(grid)
                normalized = sampler(prediction.float(), positions)
                physical = background + normalized * std + mean
                norm_err = normalized.cpu().numpy() - sample["profile_values"]
                physical_np = physical.cpu().numpy()
            _accumulate(acc[name][period], obs.cpu().numpy(), physical_np, norm_err, valid, role, platform)
            record[name] = float(np.sqrt(np.mean((physical_np - obs.cpu().numpy())[..., 0][valid[..., 0]] ** 2)))
        monthly[month] = record
        print(f"{month} done", flush=True)

    report = {name: {period: item.summary() for period, item in blocks.items()} for name, blocks in acc.items()}
    payload = {"periods": [p[0] for p in PERIODS], "models": report, "monthly_ct_rmse": monthly}
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("WROTE", args.output, flush=True)


def _period_of(month: str) -> str:
    half = "H1" if int(month[4:]) <= 6 else "H2"
    return f"{month[:4]}{half}"


def _load_role(path: str):
    from netCDF4 import Dataset
    with Dataset(path) as ds:
        return np.asarray(ds["role_code"][:]), np.asarray(ds["platform_number"][:], dtype=str)


def _accumulate(accum: Accum, obs, pred, norm_err, valid, role, platform) -> None:
    accum.add(obs, pred, norm_err, valid, role, platform)


if __name__ == "__main__":
    main()
