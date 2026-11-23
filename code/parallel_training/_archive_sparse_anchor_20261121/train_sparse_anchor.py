#!/usr/bin/env python3
"""训练/自检入口：稀疏锚点条件化的 Swin 温盐重建（原型）。

用法（服务器上，在 code 目录下）：
    PYTHONPATH=/g3240101584lt/srtp-ocean/code/python_pkgs:/g3240101584lt/srtp-ocean/code \
    /opt/conda/bin/python -u parallel_training/train_sparse_anchor.py \
        --run-dir runs/sparse_smoke --epochs 2 --max-train-months 2 --max-validation-months 1 \
        --dim 32 --blocks 2 --device cpu

说明：
  - 不修改任何现有文件；复用 server_training_data.TrainingDataset 与 train_swin_physics.Objective。
  - 损失 = 网格项 + 1.0 × 剖面查询项 + 0.01 × 物理项（与现有代码一致，便于对照）。
  - `--anchor-keep` 控制每次前向随机保留多少比例的锚点剖面，用于模拟观测密度。
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from server_training_data import TrainingDataset  # noqa: E402
from train_swin_physics import Objective, as_tensor, sample_profiles  # noqa: E402

from sparse_anchor_model import SparseAnchorModel, build_batch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=Path("/g3240101584lt/srtp-ocean/data/training_ready_v2"))
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--max-train-months", type=int, default=2)
    parser.add_argument("--max-validation-months", type=int, default=1)
    parser.add_argument("--dim", type=int, default=32)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--anchor-hidden", type=int, default=96)
    parser.add_argument("--anchor-keep", type=float, default=1.0, help="每次前向保留的锚点比例")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-query", type=float, default=1.0)
    parser.add_argument("--lambda-physics", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=20261121)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=1)
    return parser.parse_args()


@torch.no_grad()
def validate(model, dataset, months, objective, device, keep) -> dict:
    model.eval()
    ct_sse = ct_n = sa_sse = sa_n = 0.0
    for month in months:
        sample = dataset.load(month)
        batch = build_batch(sample, keep=keep)
        for key in ("grid", "values", "mask", "feats"):
            batch[key] = batch[key].to(device)
        batch["pad"] = batch["pad"].to(device)
        prediction = model(batch["grid"], batch["values"], batch["mask"], batch["feats"], batch["pad"])
        roles = torch.from_numpy(sample["profile_role_code"]).to(device)
        mask = as_tensor(sample["profile_mask"], device).bool() & (roles == 2)[:, None, None]
        profiles = sample_profiles(prediction.float(), as_tensor(sample["profile_position"], device))
        physical = objective.physical_profile_prediction(prediction.float(), sample, device)
        target = as_tensor(sample["profile_absolute"], device)
        for variable in range(2):
            variable_mask = mask[:, variable]
            error = (physical[:, variable] - target[:, variable])[variable_mask]
            if variable == 0:
                ct_sse += float((error ** 2).sum()); ct_n += int(error.numel())
            else:
                sa_sse += float((error ** 2).sum()); sa_n += int(error.numel())
        assert profiles.shape[0] == roles.shape[0]
    return {
        "CT_RMSE_C": math.sqrt(ct_sse / max(ct_n, 1)),
        "SA_RMSE_g_kg": math.sqrt(sa_sse / max(sa_n, 1)),
        "valid_points": int(ct_n + sa_n),
    }


def main() -> None:
    args = parse_args()
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    dataset = TrainingDataset(args.dataset_dir)
    train_months = [m for m, r in dataset.records.items() if r["split"] == "training"][: args.max_train_months]
    val_months = [m for m, r in dataset.records.items() if r["split"] == "validation"][: args.max_validation_months]
    if not train_months or not val_months:
        raise ValueError("需要至少一个训练月和一个验证月")

    probe = dataset.load(train_months[0])
    probe_batch = build_batch(probe)
    in_channels = probe_batch["grid"].shape[1]
    anchors = probe_batch["values"].shape[1]
    model = SparseAnchorModel(
        in_channels, args.dim, args.blocks, args.heads, args.window, args.anchor_hidden
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    objective = Objective(dataset, device, args.lambda_query, args.lambda_physics, 1e-4, 0.01)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "run_dir": str(args.run_dir), "in_channels": int(in_channels),
              "anchors_probe": int(anchors),
              "parameters": sum(p.numel() for p in model.parameters())}
    config["dataset_dir"] = str(args.dataset_dir)
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"in_channels": in_channels, "anchors": anchors,
                      "parameters": config["parameters"], "device": str(device)}), flush=True)

    rng = torch.Generator().manual_seed(args.seed)
    best = math.inf
    log_path = args.run_dir / "train.log"
    for epoch in range(1, args.epochs + 1):
        model.train()
        started = time.time()
        order = train_months.copy()
        random.Random(args.seed + epoch).shuffle(order)
        totals = {"total": 0.0, "grid": 0.0, "query": 0.0, "physics": 0.0}
        for step, month in enumerate(order, 1):
            sample = dataset.load(month)
            batch = build_batch(sample, keep=args.anchor_keep, generator=rng)
            for key in ("grid", "values", "mask", "feats"):
                batch[key] = batch[key].to(device)
            batch["pad"] = batch["pad"].to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch["grid"], batch["values"], batch["mask"], batch["feats"], batch["pad"])
            loss, parts = objective(prediction, sample, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            totals["total"] += float(loss.detach())
            for name, value in parts.items():
                totals[name] += float(value.detach())
            if step % args.log_every == 0 or step == len(order):
                line = (f"epoch={epoch} step={step}/{len(order)} month={month} "
                        f"anchors={batch['values'].shape[1]} loss={float(loss):.6f}")
                print(line, flush=True)
        metrics = validate(model, dataset, val_months, objective, device, args.anchor_keep)
        record = {"epoch": epoch, "seconds": round(time.time() - started, 2),
                  **{f"train_{k}": v / len(order) for k, v in totals.items()},
                  **{f"validation_{k}": v for k, v in metrics.items()}}
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        print(json.dumps(record), flush=True)
        score = metrics["CT_RMSE_C"] + metrics["SA_RMSE_g_kg"]
        if score < best:
            best = score
            torch.save({"model": model.state_dict(), "config": config, "epoch": epoch},
                       args.run_dir / "best.pt")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
