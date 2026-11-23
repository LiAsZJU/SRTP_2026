#!/usr/bin/env python3
"""v4（方案 A）四模型 × 半年节点评估：2020H1–2025H1，E 组 Argo 剖面。

注意：必须带 ABLATE_MODE=no_smap 与 ablate_site 的 PYTHONPATH 运行，保证输入规格与训练一致：
  PYTHONPATH=ablate_site:python_pkgs:code:parallel_training ABLATE_MODE=no_smap \
  python dataset_v2/halfyear_eval_v4.py --suite <v4A_four_models_...> --output <out.json>
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
for p in (str(CODE_DIR), str(CODE_DIR / "parallel_training")):
    if p not in sys.path:
        sys.path.insert(0, p)

from dataset_v2.grid_sampler import GridSampler  # noqa: E402
from residual_cnn.model import ResidualCNN  # noqa: E402
from server_training_data import TrainingDataset  # noqa: E402
from train_swin_physics import Objective, SwinReconstructor, as_tensor, model_input  # noqa: E402
from unet_baseline.model import UNetReconstructor  # noqa: E402
from vit_baseline.model import ViTReconstructor  # noqa: E402

DATASET = Path("/g3240101584lt/srtp-ocean/data/training_ready_v4")
PERIODS = [("2020H1", "202001", "202006"), ("2020H2", "202007", "202012"),
           ("2021H1", "202101", "202106"), ("2021H2", "202107", "202112"),
           ("2022H1", "202201", "202206"), ("2022H2", "202207", "202212"),
           ("2023H1", "202301", "202306"), ("2023H2", "202307", "202312"),
           ("2024H1", "202401", "202406"), ("2024H2", "202407", "202412"),
           ("2025H1", "202501", "202506")]
ARCHS = {"swin": "swin", "cnn": "cnn", "unet": "unet", "vit": "vit"}


def build(arch: str, config: dict) -> torch.nn.Module:
    if arch == "swin":
        return SwinReconstructor(276, config["dim"], config["blocks"], config["heads"], config["window"])
    if arch == "cnn":
        return ResidualCNN(width=config["dim"], blocks=config["blocks"])
    if arch == "unet":
        return UNetReconstructor(width=config["dim"])
    return ViTReconstructor(embedding_dim=config["dim"], blocks=config["blocks"], heads=config["heads"], dropout=0.0)


class Accum:
    def __init__(self) -> None:
        self.v = np.zeros((2, 8), dtype="f8")  # count, sum, sumsq, esum, sse, nse(norm), _, _
        self.profiles = 0
        self.platforms: set[str] = set()

    def add(self, obs, pred, norm_err, valid, role, platform) -> None:
        cohort = role == 2
        good = cohort & valid.any(axis=(1, 2))
        self.profiles += int(good.sum())
        self.platforms.update(platform[good].tolist())
        for i in range(2):
            mask = cohort[:, None] & valid[:, i]
            if not mask.any():
                continue
            y = obs[:, i][mask].astype("f8")
            e = pred[:, i][mask].astype("f8") - y
            ne = norm_err[:, i][mask].astype("f8")
            self.v[i] += (y.size, y.sum(), (y * y).sum(), e.sum(), (e * e).sum(), (ne * ne).sum(), 0, 0)

    def summary(self) -> dict:
        out = {"profiles": self.profiles, "platforms": len(self.platforms)}
        total = float(self.v[:, 0].sum())
        out["normalized_score"] = math.sqrt(self.v[:, 5].sum() / total) if total else None
        for i, name in enumerate(("CT", "SA")):
            c, t, sq, es, sse, *_ = self.v[i]
            out[name] = None if c == 0 else {
                "points": int(c), "rmse": math.sqrt(sse / c), "bias": es / c,
                "r2": 1 - sse / (sq - t * t / c) if (sq - t * t / c) > 0 else None,
            }
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
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
    for arch in ARCHS:
        run = args.suite / arch
        ckpt = torch.load(run / "best.pt", map_location="cpu", weights_only=False)
        model = build(arch, ckpt["config"]).to(device)
        model.load_state_dict(ckpt["model"])
        model.eval()
        models[arch] = model
        print("loaded", arch, "epoch", ckpt.get("epoch"), flush=True)

    acc = {name: {p[0]: Accum() for p in PERIODS} for name in list(models) + ["climatology"]}
    for name, start, end in PERIODS:
        y, mo = int(start[:4]), int(start[4:])
        while f"{y:04d}{mo:02d}" <= end:
            month = f"{y:04d}{mo:02d}"
            mo += 1
            if mo > 12:
                y, mo = y + 1, 1
            if month not in source.records:
                continue
            sample = source.load(month)
            from netCDF4 import Dataset
            with Dataset(source.records[month]["argo_profiles"]) as ds:
                role = np.asarray(ds["role_code"][:])
                platform = np.asarray(ds["platform_number"][:], dtype=str)
            valid = sample["profile_mask"].astype(bool)
            obs = as_tensor(sample["profile_absolute"], device).cpu().numpy()
            positions = as_tensor(sample["profile_position"], device)
            background = as_tensor(sample["profile_background"], device)
            grid = model_input(sample, device)
            _add(acc["climatology"][name], obs, background.cpu().numpy(),
                 np.zeros_like(obs), valid, role, platform)
            for arch, model in models.items():
                with torch.no_grad():
                    normalized = sampler(model(grid).float(), positions)
                    physical = (background + normalized * std + mean).cpu().numpy()
                    norm_err = normalized.cpu().numpy() - sample["profile_values"]
                _add(acc[arch][name], obs, physical, norm_err, valid, role, platform)
            print(month, "ok", flush=True)

    payload = {
        "suite": str(args.suite),
        "periods": [p[0] for p in PERIODS],
        "models": {name: {p: blk.summary() for p, blk in blocks.items()} for name, blocks in acc.items()},
    }
    args.output.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print("WROTE", args.output, flush=True)


def _add(accum: Accum, obs, pred, norm_err, valid, role, platform) -> None:
    accum.add(obs, pred, norm_err, valid, role, platform)


if __name__ == "__main__":
    main()
