#!/usr/bin/env python3
"""评估稀疏锚点模型：在 test_locked / forward_test 的留出 E 组剖面上算指标。

输出结构与 dataset_v2/evaluate_checkpoint.py 保持一致（同一套 Metrics / cohort），
方便与 276 通道稠密模型的结果直接对比。

用法：
    cd /g3240101584lt/srtp-ocean/code
    export PYTHONPATH=$PWD/python_pkgs:$PWD
    /opt/conda/bin/python -u parallel_training/evaluate_sparse_anchor.py \
        --checkpoint runs/sparse_anchor_200ep_gpu_xxx/best.pt \
        --output runs/sparse_anchor_200ep_gpu_xxx/test_metrics.json \
        --splits test_locked forward_test
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset_v2.evaluate_checkpoint import Metrics, eligible_validation_platforms, load_profiles  # noqa: E402
from dataset_v2.grid_sampler import GridSampler  # noqa: E402
from server_training_data import TrainingDataset  # noqa: E402
from sparse_anchor_model import SparseAnchorModel, anchors_from_sample, grid_input  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, default=Path("/g3240101584lt/srtp-ocean/data/training_ready_v2"))
    p.add_argument("--splits", nargs="+", default=["test_locked", "forward_test"])
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--anchor-keep", type=float, default=1.0)
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    source = TrainingDataset(args.dataset_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint["config"]
    sampler = GridSampler(Path(next(iter(source.records.values()))["background"]), device)
    model = SparseAnchorModel(config["in_channels"], config["dim"], config["blocks"],
                              config["heads"], config["window"], config["anchor_hidden"]).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    mean = torch.as_tensor(np.stack((source.statistics["target.CT_anomaly"]["mean"],
                                     source.statistics["target.SA_anomaly"]["mean"])),
                           dtype=torch.float32, device=device).reshape(1, 2, -1, 1, 1)
    std = torch.as_tensor(np.stack((source.statistics["target.CT_anomaly"]["std"],
                                    source.statistics["target.SA_anomaly"]["std"])),
                          dtype=torch.float32, device=device).reshape(1, 2, -1, 1, 1)

    validation_platforms = sorted(eligible_validation_platforms(source))
    report = {"checkpoint": str(args.checkpoint.resolve()), "epoch": checkpoint["epoch"],
              "dataset": str(args.dataset_dir.resolve()), "device": str(device),
              "anchor_keep": args.anchor_keep, "metrics": {}}

    for split in args.splits:
        months = [m for m, row in source.records.items() if row["split"] == split]
        if not months:
            raise ValueError(f"empty split {split}")
        metrics = {"all_E": Metrics(), "E_platforms_unseen_in_validation": Metrics()}
        for month in months:
            sample = source.load(month)
            role, platform = load_profiles(source.records[month]["argo_profiles"])
            anchors = anchors_from_sample(sample)
            keep = args.anchor_keep
            if keep < 1.0:
                n = anchors["values"].shape[0]
                k = max(1, int(round(n * keep)))
                idx = torch.randperm(n)[:k].sort().values
                anchors = {key: anchors[key][idx] for key in ("values", "mask", "feats")}
            grid = grid_input(sample).to(device)
            values = anchors["values"].to(device)[None]
            amask = anchors["mask"].to(device)[None]
            feats = anchors["feats"].to(device)[None]
            pad = torch.zeros(1, values.shape[1], dtype=torch.bool, device=device)
            prediction = model(grid, values, amask, feats, pad).float()

            positions = torch.as_tensor(sample["profile_position"], dtype=torch.float32, device=device)
            background = torch.as_tensor(sample["profile_background"], dtype=torch.float32, device=device)
            normalized = sampler(prediction, positions)
            physical = background + normalized * std[0, :, :, 0, 0][None] + mean[0, :, :, 0, 0][None]
            errors = normalized.cpu().numpy() - sample["profile_values"]
            physical = physical.cpu().numpy()
            valid = sample["profile_mask"].astype(bool)
            cohorts = {
                "all_E": role == 2,
                "E_platforms_unseen_in_validation": (role == 2) & ~np.isin(platform, validation_platforms),
            }
            for name, cohort in cohorts.items():
                metrics[name].add(sample["profile_absolute"], physical, errors, valid, cohort, platform)
        report["metrics"][split] = {"months": months,
                                    "operators": {"native_coordinates":
                                                  {name: acc.summary() for name, acc in metrics.items()}}}
        summary = {name: acc.summary() for name, acc in metrics.items()}
        print(split, json.dumps(summary), flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("WROTE", args.output, flush=True)


if __name__ == "__main__":
    main()
