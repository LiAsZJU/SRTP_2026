#!/usr/bin/env python3
"""物理一致性诊断：对给定 checkpoint 计算预测场与参考场的“垂向稳定”违规率。

用途：配合 A1（λ 扫描）判断物理约束到底有没有把违规压下去。
输出 JSON：每个 split 的预测/参考违规率、平均违规幅度。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import sys
CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from server_training_data import TrainingDataset  # noqa: E402
from train_swin_physics import Objective, SwinReconstructor, as_tensor, model_input  # noqa: E402


def violation_stats(physical_ct: torch.Tensor, physical_sa: torch.Tensor, alpha: torch.Tensor,
                    beta: torch.Tensor, mask: torch.Tensor, scale: float, tol: float) -> dict:
    alpha_mid = 0.5 * (alpha[:, 1:] + alpha[:, :-1])
    beta_mid = 0.5 * (beta[:, 1:] + beta[:, :-1])
    proxy = (beta_mid * (physical_sa[:, 1:] - physical_sa[:, :-1])
             - alpha_mid * (physical_ct[:, 1:] - physical_ct[:, :-1])) / scale
    valid = mask[:, :, 1:] * mask[:, :, :-1]
    valid = valid[:, 0] * valid[:, 1] * (alpha_mid != 0) * (beta_mid != 0)
    valid_bool = valid.bool()
    if not valid_bool.any():
        return {"valid_pairs": 0, "violation_rate": None, "mean_violation": None}
    v = proxy[valid_bool]
    viol = torch.relu(-v - tol)
    return {
        "valid_pairs": int(valid_bool.sum()),
        "violation_rate": float((viol > 0).float().mean()),
        "mean_proxy": float(v.mean()),
        "violation_rate_0": float((v < 0).float().mean()),
        "mean_violation": float(viol[viol > 0].mean()) if (viol > 0).any() else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("/g3240101584lt/srtp-ocean/data/training_ready_v2"))
    parser.add_argument("--splits", nargs="+", default=["validation"])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)

    source = TrainingDataset(args.dataset_dir)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    model = SwinReconstructor(config["input_channels"], config["dim"], config["blocks"],
                              config["heads"], config["window"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    objective = Objective(source, device, 1.0, 0.01, config["stability_scale"], config["stability_tolerance"])

    out = {"checkpoint": str(args.checkpoint), "epoch": ckpt.get("epoch"), "lambda_physics": config.get("lambda_physics")}
    for split in args.splits:
        months = [m for m, r in source.records.items() if r["split"] == split]
        pred_stats, ref_stats = [], []
        for month in months:
            sample = source.load(month)
            with torch.no_grad():
                prediction = model(model_input(sample, device)).float()
            background = as_tensor(sample["background"][None], device)
            physical = background + prediction * objective.std + objective.mean
            ref = as_tensor(sample["target_absolute"][None], device)
            mask = as_tensor(sample["target_mask"][None], device)
            alpha = as_tensor(sample["alpha"][None], device)
            beta = as_tensor(sample["beta"][None], device)
            pred_stats.append(violation_stats(physical[:, 0], physical[:, 1], alpha, beta, mask,
                                              config["stability_scale"], config["stability_tolerance"]))
            ref_stats.append(violation_stats(ref[:, 0], ref[:, 1], alpha, beta, mask,
                                             config["stability_scale"], config["stability_tolerance"]))
        def agg(rows: list[dict]) -> dict:
            pairs = sum(r["valid_pairs"] for r in rows)
            if not pairs:
                return {"valid_pairs": 0}
            return {
                "valid_pairs": pairs,
                "violation_rate": float(sum((r["violation_rate"] or 0) * r["valid_pairs"] for r in rows) / pairs),
                "violation_rate_0": float(sum((r["violation_rate_0"] or 0) * r["valid_pairs"] for r in rows) / pairs),
                "mean_violation": float(np.mean([r["mean_violation"] or 0 for r in rows])),
            }
        out[split] = {"prediction": agg(pred_stats), "reference_CORA": agg(ref_stats), "months": len(months)}
    args.output.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
