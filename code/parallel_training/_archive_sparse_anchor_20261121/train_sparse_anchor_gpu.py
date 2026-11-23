#!/usr/bin/env python3
"""GPU 版：稀疏 Argo 锚点 token 条件化的 Swin 温盐重建（200 轮实验用）。

与 train_sparse_anchor.py 的区别：
  - 启动时把 80 训练月 + 24 验证月一次性搬进显存缓存（约 2 GiB），之后每轮不再读盘；
  - 损失在张量侧直接计算（不经过 numpy），与 train_parallel.py 的 grid/query/physics 口径一致；
  - 支持 --resume、best.pt/latest.pt 原子写、metrics.jsonl；
  - 训练结束后在验证集上按多档锚点保留率评估，直接产出“信息曲线”。

用法：
    cd /g3240101584lt/srtp-ocean/code
    export PYTHONPATH=$PWD/python_pkgs:$PWD
    /opt/conda/bin/python -u parallel_training/train_sparse_anchor_gpu.py \
        --run-dir /g3240101584lt/srtp-ocean/runs/sparse_anchor_200ep_gpu_20261121 \
        --epochs 200 --seed 20261120 --dim 48 --blocks 4
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from dataset_v2.grid_sampler import GridSampler  # noqa: E402
from server_training_data import TrainingDataset  # noqa: E402
from train_swin_physics import SwinBlock  # noqa: E402

from sparse_anchor_model import SparseAnchorModel, anchors_from_sample, grid_input  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--dataset-dir", type=Path, default=Path("/g3240101584lt/srtp-ocean/data/training_ready_v2"))
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--patience", type=int, default=201)
    p.add_argument("--dim", type=int, default=48)
    p.add_argument("--blocks", type=int, default=4)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--window", type=int, default=8)
    p.add_argument("--anchor-hidden", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-query", type=float, default=1.0)
    p.add_argument("--lambda-physics", type=float, default=0.01)
    p.add_argument("--stability-scale", type=float, default=1e-4)
    p.add_argument("--stability-tolerance", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=20261120)
    p.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-train-months", type=int)
    p.add_argument("--max-validation-months", type=int)
    p.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--eval-keeps", type=float, nargs="+", default=[1.0, 0.5, 0.25, 0.1, 0.05])
    return p.parse_args()


def atomic_save(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


class CachedMonth:
    """一个月样本在显存里的张量（避免每轮重复读 NetCDF / 重复 to(device)）。"""

    def __init__(self, sample: dict, grid: np.ndarray, lat: np.ndarray, lon: np.ndarray,
                 anchors: dict, mean: np.ndarray, std: np.ndarray, device: torch.device, area_full: np.ndarray) -> None:
        def t(a, dtype=torch.float32):
            return torch.as_tensor(np.asarray(a), dtype=dtype, device=device)

        self.grid = t(grid)[None]
        self.target = t(sample["target"])[None]
        self.target_mask = t(sample["target_mask"])[None]
        self.background = t(sample["background"])[None]
        self.alpha = t(sample["alpha"])[None]
        self.beta = t(sample["beta"])[None]
        self.area = t(area_full)[None, None, None]
        self.positions = t(sample["profile_position"])
        self.profile_values = t(sample["profile_values"])
        self.profile_mask = t(sample["profile_mask"])
        self.role = torch.as_tensor(np.asarray(sample["profile_role_code"]), dtype=torch.int64, device=device)
        self.profile_absolute = t(sample["profile_absolute"])
        self.profile_background = t(sample["profile_background"])
        self.anchor_values = t(anchors["values"])
        self.anchor_mask = t(anchors["mask"])
        self.anchor_feats = t(anchors["feats"])
        self.mean = t(mean.reshape(1, 2, -1, 1, 1))
        self.std = t(std.reshape(1, 2, -1, 1, 1))
        self.normalized_values = self.profile_values  # 已归一化

    def anchors(self, keep: float, rng: torch.Generator | None = None) -> tuple:
        v, m, f = self.anchor_values, self.anchor_mask, self.anchor_feats
        n = v.shape[0]
        if keep < 1.0 and n > 1:
            k = max(1, int(round(n * keep)))
            idx = torch.randperm(n, generator=rng, device=v.device)[:k].sort().values
            v, m, f = v[idx], m[idx], f[idx]
        return v[None], m[None], f[None], torch.zeros(1, v.shape[0], dtype=torch.bool, device=v.device)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    effective = mask if weights is None else mask * weights
    return (values * effective).sum() / effective.sum().clamp_min(1.0)


def train_loss(prediction, month: CachedMonth, args, sampler: GridSampler, device) -> tuple:
    grid_loss = masked_mean((prediction - month.target).square(), month.target_mask, month.area)

    profiles = sampler(prediction, month.positions)                       # (P, 2, 60)
    query_mask = month.profile_mask * (month.role == 1)[:, None, None]
    query_error = F.smooth_l1_loss(profiles, month.profile_values, reduction="none", beta=0.5)
    query_loss = masked_mean(query_error, query_mask)

    physical = month.background + prediction * month.std + month.mean
    ct, sa = physical[:, 0], physical[:, 1]
    alpha_mid = 0.5 * (month.alpha[:, 1:] + month.alpha[:, :-1])
    beta_mid = 0.5 * (month.beta[:, 1:] + month.beta[:, :-1])
    proxy = (beta_mid * (sa[:, 1:] - sa[:, :-1]) - alpha_mid * (ct[:, 1:] - ct[:, :-1])) / args.stability_scale
    valid = month.target_mask[:, :, 1:] * month.target_mask[:, :, :-1]
    valid = valid[:, 0] * valid[:, 1] * (alpha_mid != 0) * (beta_mid != 0)
    physics = masked_mean(F.relu(-proxy - args.stability_tolerance).square(), valid, month.area[:, 0, 0])

    total = grid_loss + args.lambda_query * query_loss + args.lambda_physics * physics
    return total, {"grid": float(grid_loss.detach()), "query": float(query_loss.detach()),
                   "physics": float(physics.detach())}


@torch.no_grad()
def validate(model, months, sampler, device, use_amp: bool, keep: float = 1.0) -> dict:
    model.eval()
    norm_sum = norm_count = 0.0
    phys_sse = np.zeros(2, dtype="f8")
    phys_count = np.zeros(2, dtype="i8")
    profiles_seen = 0
    for month in months:
        v, m, f, pad = month.anchors(keep)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            prediction = model(month.grid, v, m, f, pad)
        prediction = prediction.float()
        sampled = sampler(prediction, month.positions)
        mask = month.profile_mask.bool() & (month.role == 2)[:, None, None]
        norm_sum += float(torch.where(mask, (sampled - month.profile_values).square(), 0).sum())
        norm_count += float(mask.sum())
        physical = month.profile_background + sampled * month.std[0, :, :, 0, 0][None] + month.mean[0, :, :, 0, 0][None]
        for var in range(2):
            vm = mask[:, var]
            err = physical[:, var] - month.profile_absolute[:, var]
            phys_sse[var] += float((err[vm] ** 2).sum())
            phys_count[var] += int(vm.sum())
        profiles_seen += int(mask.any(dim=(1, 2)).sum())
    model.train()
    return {
        "score": math.sqrt(norm_sum / max(norm_count, 1)),
        "CT_RMSE_C": math.sqrt(phys_sse[0] / max(phys_count[0], 1)),
        "SA_RMSE_g_kg": math.sqrt(phys_sse[1] / max(phys_count[1], 1)),
        "valid_points": int(phys_count.sum()),
        "profiles": profiles_seen,
    }


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA 不可用")

    dataset = TrainingDataset(args.dataset_dir)
    train_months = [mm for mm, r in dataset.records.items() if r["split"] == "training"]
    val_months = [mm for mm, r in dataset.records.items() if r["split"] == "validation"]
    if args.max_train_months:
        train_months = train_months[: args.max_train_months]
    if args.max_validation_months:
        val_months = val_months[: args.max_validation_months]

    reference = dataset.records[train_months[0]]["background"]
    sampler = GridSampler(Path(reference), device)
    stats = dataset.statistics
    mean = np.stack((stats["target.CT_anomaly"]["mean"], stats["target.SA_anomaly"]["mean"]))
    std = np.stack((stats["target.CT_anomaly"]["std"], stats["target.SA_anomaly"]["std"]))
    from netCDF4 import Dataset as NC
    with NC(reference) as ds:
        lat = np.asarray(ds["latitude"][:], dtype="f4")
        lon = np.asarray(ds["longitude"][:], dtype="f4")
    area_full = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], (len(lat), len(lon))).astype("f4")

    print(json.dumps({"device": str(device), "train_months": len(train_months),
                      "validation_months": len(val_months)}), flush=True)
    t0 = time.time()
    train_cache = []
    for i, month in enumerate(train_months, 1):
        sample = dataset.load(month)
        train_cache.append(CachedMonth(sample, grid_input(sample)[0].numpy(), lat, lon,
                                       anchors_from_sample(sample), mean, std, device, area_full))
        if i % 20 == 0:
            print(json.dumps({"cached_train": i, "elapsed_s": round(time.time() - t0, 1)}), flush=True)
    val_cache = []
    for month in val_months:
        sample = dataset.load(month)
        val_cache.append(CachedMonth(sample, grid_input(sample)[0].numpy(), lat, lon,
                                     anchors_from_sample(sample), mean, std, device, area_full))
    print(json.dumps({"cache_built_s": round(time.time() - t0, 1)}), flush=True)

    in_channels = train_cache[0].grid.shape[1]
    model = SparseAnchorModel(in_channels, args.dim, args.blocks, args.heads, args.window,
                              args.anchor_hidden).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    use_amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {**vars(args), "run_dir": str(args.run_dir), "dataset_dir": str(args.dataset_dir),
              "in_channels": int(in_channels), "parameters": sum(p.numel() for p in model.parameters()),
              "anchors_per_month_probe": int(train_cache[0].anchor_values.shape[0])}
    config["run_dir"] = str(config["run_dir"])
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"parameters": config["parameters"], "anchors": config["anchors_per_month_probe"]}), flush=True)

    start_epoch, best, stale = 1, math.inf, 0
    latest = args.run_dir / "latest.pt"
    if args.resume and latest.exists():
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best, stale = ckpt["best_score"], ckpt["stale"]
        print(json.dumps({"resumed_from_epoch": start_epoch - 1}), flush=True)

    metrics_path = args.run_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        model.train()
        order = list(range(len(train_cache)))
        random.Random(args.seed + epoch).shuffle(order)
        totals = {"total": 0.0, "grid": 0.0, "query": 0.0, "physics": 0.0}
        for step, index in enumerate(order, 1):
            month = train_cache[index]
            v, m, f, pad = month.anchors(1.0)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                prediction = model(month.grid, v, m, f, pad)
                loss, parts = train_loss(prediction, month, args, sampler, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            totals["total"] += float(loss.detach())
            for k in ("grid", "query", "physics"):
                totals[k] += parts[k]
            if step % args.log_every == 0:
                print(f"epoch={epoch} step={step}/{len(order)} loss={float(loss):.6f}", flush=True)
        metrics = validate(model, val_cache, sampler, device, use_amp)
        improved = metrics["score"] < best
        best, stale = (metrics["score"], 0) if improved else (best, stale + 1)
        record = {"epoch": epoch, "seconds": round(time.time() - started, 2),
                  **{f"train_{k}": v / len(order) for k, v in totals.items()},
                  **{f"validation_{k}": v for k, v in metrics.items()}, "best": improved}
        with metrics_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record) + "\n")
        ckpt = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(), "best_score": best, "stale": stale, "config": config}
        atomic_save(latest, ckpt)
        if improved:
            atomic_save(args.run_dir / "best.pt", ckpt)
        print(json.dumps(record), flush=True)
        if stale >= args.patience:
            print(f"early_stop epoch={epoch}", flush=True)
            break

    # 训练结束后：固定 best 权重，扫锚点保留率 → 信息曲线
    best_path = args.run_dir / "best.pt"
    if best_path.exists():
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=False)["model"])
    curve = {}
    for keep in args.eval_keeps:
        curve[str(keep)] = validate(model, val_cache, sampler, device, use_amp, keep=keep)
        print(json.dumps({"info_curve": keep, **curve[str(keep)]}), flush=True)
    (args.run_dir / "info_curve.json").write_text(json.dumps(curve, indent=2) + "\n", encoding="utf-8")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
