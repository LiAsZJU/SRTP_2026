#!/usr/bin/env python3
"""逐剖面 / 逐深度误差诊断：定位 2025 年误差上升的空间与深度来源。

输出：
  <prefix>_profiles.csv   每行 = 一个月 × 一个模型 × 一条 E 剖面
  <prefix>_depth.json     每个 (月份, 模型, 变量) 的逐深度 RMSE / bias
"""

from __future__ import annotations

import argparse
import csv
import json
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

DATASET = Path("/g3240101584lt/srtp-ocean/data/training_ready_v2")
SUITE = Path("/g3240101584lt/srtp-ocean/runs/four_models_200ep_seed20261120_20261120T115114Z")
RUNS = {
    "swin200": (SUITE / "swin", "swin"),
    "cnn200": (SUITE / "residual_cnn", "cnn"),
}


def build(kind: str, config: dict) -> torch.nn.Module:
    if kind == "swin":
        return SwinReconstructor(276, config["dim"], config["blocks"], config["heads"], config["window"])
    return ResidualCNN(config["width"], config["blocks"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--months", nargs="+", required=True)
    parser.add_argument("--prefix", type=Path, required=True)
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device("cpu")

    source = TrainingDataset(DATASET)
    reference = next(iter(source.records.values()))
    sampler = GridSampler(Path(reference["background"]), device)
    objective = Objective(source, device, 1.0, 0.01, 1e-4, 0.01)
    mean = objective.mean[0, :, :, 0, 0]
    std = objective.std[0, :, :, 0, 0]

    models = {}
    for name, (run_dir, kind) in RUNS.items():
        ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=False)
        model = build(kind, ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[name] = model

    depth_stats: dict[str, dict] = {}
    rows = []
    for month in args.months:
        sample = source.load(month)
        with __import__("netCDF4").Dataset(source.records[month]["argo_profiles"]) as ds:
            role = np.asarray(ds["role_code"][:])
            platform = np.asarray(ds["platform_number"][:], dtype=str)
        valid = sample["profile_mask"].astype(bool)
        obs = as_tensor(sample["profile_absolute"], device)
        positions = as_tensor(sample["profile_position"], device)
        background = as_tensor(sample["profile_background"], device)
        grid = model_input(sample, device)
        depth = np.arange(60)
        for name, model in models.items():
            with torch.no_grad():
                prediction = model(grid)
                normalized = sampler(prediction.float(), positions)
                physical = (background + normalized * std + mean).cpu().numpy()
            err = physical - obs.cpu().numpy()
            key = f"{month}|{name}"
            per_depth = {}
            for i, var in enumerate(("CT", "SA")):
                mask = (role == 2)[:, None] & valid[:, i]
                sse = np.where(mask, err[:, i] ** 2, 0).sum(axis=0)
                cnt = mask.sum(axis=0)
                bias = np.where(mask, err[:, i], 0).sum(axis=0) / np.maximum(cnt, 1)
                per_depth[var] = {
                    "rmse": [None if c == 0 else float(np.sqrt(s / c)) for s, c in zip(sse, cnt)],
                    "bias": [None if c == 0 else float(b) for b, c in zip(bias, cnt)],
                    "count": cnt.tolist(),
                }
            depth_stats[key] = per_depth
            for p in range(len(role)):
                if role[p] != 2:
                    continue
                entry = {
                    "month": month,
                    "model": name,
                    "profile": p,
                    "platform": str(platform[p]),
                    "lat": float(sample["profile_position"][p, 0]),
                    "lon": float(sample["profile_position"][p, 1]),
                    "time_offset": float(sample["profile_position"][p, 2]),
                }
                for i, var in enumerate(("CT", "SA")):
                    m = valid[p, i]
                    if m.any():
                        e = err[p, i][m]
                        entry[f"{var.lower()}_rmse"] = float(np.sqrt(np.mean(e ** 2)))
                        entry[f"{var.lower()}_bias"] = float(np.mean(e))
                        entry[f"{var.lower()}_levels"] = int(m.sum())
                    else:
                        entry[f"{var.lower()}_rmse"] = None
                        entry[f"{var.lower()}_bias"] = None
                        entry[f"{var.lower()}_levels"] = 0
                rows.append(entry)
        print(month, "ok", flush=True)

    prefix = args.prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_name(prefix.name + "_profiles.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    prefix.with_name(prefix.name + "_depth.json").write_text(json.dumps(depth_stats), encoding="utf-8")
    print("WROTE", prefix, len(rows), flush=True)


if __name__ == "__main__":
    main()
