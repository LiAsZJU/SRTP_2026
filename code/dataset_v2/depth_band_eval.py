#!/usr/bin/env python3
"""全时段 × 分深度带评估：Swin vs CNN 的垂向性能对比（含多种子）。

输出：<prefix>.json
  models[model][period][band][var] = {rmse, bias, points}
  以及 models[model]["ALL"][band][var]
"""

from __future__ import annotations

import argparse
import json
import keyword
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
SUITE20 = Path("/g3240101584lt/srtp-ocean/runs/four_models_200ep_seed20261120_20261120T115114Z")
SUITE17 = Path("/g3240101584lt/srtp-ocean/runs/baseline_suite_v2_20261120T105835Z")
RUNS = {
    "swin500_s17": (Path("/g3240101584lt/srtp-ocean/runs/swin_physics_v2_500ep_gpu_20261118T044643Z"), "swin"),
    "swin200_s20": (SUITE20 / "swin", "swin"),
    "cnn250_s17": (SUITE17 / "residual_cnn/seed_20261117", "cnn"),
    "cnn250_s18": (SUITE17 / "residual_cnn/seed_20261118", "cnn"),
    "cnn200_s20": (SUITE20 / "residual_cnn", "cnn"),
}
BANDS = [(0, 10, "5-50"), (10, 20, "50-100"), (20, 30, "100-200"), (30, 40, "200-300"), (40, 60, "300-500")]
PERIODS = [("2022H1", "202201", "202206"), ("2022H2", "202207", "202212"),
           ("2023H1", "202301", "202306"), ("2023H2", "202307", "202312"),
           ("2024H1", "202401", "202406"), ("2024H2", "202407", "202412"),
           ("2025H1", "202501", "202506")]


def build(kind: str, config: dict):
    if kind == "swin":
        return SwinReconstructor(276, config["dim"], config["blocks"], config["heads"], config["window"])
    return ResidualCNN(config["width"], config["blocks"])


class BandAcc:
    def __init__(self) -> None:
        self.data = {(band[2], var): [0, 0.0, 0.0] for band in BANDS for var in ("CT", "SA")}

    def add(self, obs, pred, valid, role):
        cohort = role == 2
        for i, var in enumerate(("CT", "SA")):
            mask = cohort[:, None] & valid[:, i]
            if not mask.any():
                continue
            err = pred[:, i] - obs[:, i]
            for lo, hi, label in BANDS:
                m = mask[:, lo:hi]
                e = err[:, lo:hi][m]
                if e.size:
                    slot = self.data[(label, var)]
                    slot[0] += e.size
                    slot[1] += float(e.sum())
                    slot[2] += float((e ** 2).sum())

    def summary(self):
        out = {}
        for (label, var), (n, esum, sse) in self.data.items():
            out.setdefault(label, {})[var] = (
                None if n == 0 else {"points": int(n), "rmse": float(np.sqrt(sse / n)), "bias": esum / n}
            )
        return out


def main() -> None:
    parser = argparse.ArgumentParser()
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

    acc = {name: {p[0]: BandAcc() for p in PERIODS} for name in RUNS}
    acc_all = {name: BandAcc() for name in RUNS}

    months = [m for m in source.records if "202201" <= m <= "202506"]
    for month in sorted(months):
        period = f"{month[:4]}{'H1' if int(month[4:]) <= 6 else 'H2'}"
        sample = source.load(month)
        from netCDF4 import Dataset
        with Dataset(source.records[month]["argo_profiles"]) as ds:
            role = np.asarray(ds["role_code"][:])
        valid = sample["profile_mask"].astype(bool)
        obs = as_tensor(sample["profile_absolute"], device).cpu().numpy()
        positions = as_tensor(sample["profile_position"], device)
        background = as_tensor(sample["profile_background"], device)
        grid = model_input(sample, device)
        for name, model in models.items():
            with torch.no_grad():
                normalized = sampler(model(grid).float(), positions)
                physical = (background + normalized * std + mean).cpu().numpy()
            acc[name][period].add(obs, physical, valid, role)
            acc_all[name].add(obs, physical, valid, role)
        print(month, "ok", flush=True)

    payload = {
        "periods": [p[0] for p in PERIODS],
        "bands": [b[2] for b in BANDS],
        "models": {name: {p: blk.summary() for p, blk in blocks.items()} for name, blocks in acc.items()},
        "all": {name: acc_all[name].summary() for name in RUNS},
    }
    args.prefix.parent.mkdir(parents=True, exist_ok=True)
    args.prefix.write_text(json.dumps(payload, indent=1, ensure_ascii=False), encoding="utf-8")
    print("WROTE", args.prefix, flush=True)


if __name__ == "__main__":
    main()
