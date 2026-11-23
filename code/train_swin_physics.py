#!/usr/bin/env python3
"""Train a compact 2-D Swin model for monthly 3-D CT/SA reconstruction."""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import ShiftedWindowAttention

from server_training_data import TrainingDataset


DEFAULT_RUN_DIR = Path("/g3240101584lt/srtp-ocean/runs/swin_physics_v1")


class SwinBlock(nn.Module):
    def __init__(self, dim: int, heads: int, window: int, shifted: bool):
        super().__init__()
        shift = window // 2 if shifted else 0
        self.norm1 = nn.LayerNorm(dim)
        self.attention = ShiftedWindowAttention(
            dim=dim, window_size=[window, window], shift_size=[shift, shift], num_heads=heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class SwinReconstructor(nn.Module):
    """Patch size 3 exactly maps 81x321 to 27x107 and back."""
    def __init__(self, in_channels: int, dim: int = 48, blocks: int = 4,
                 heads: int = 4, window: int = 8):
        super().__init__()
        if dim % heads:
            raise ValueError("Embedding dimension must be divisible by attention heads")
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=3, stride=3)
        self.blocks = nn.ModuleList(
            SwinBlock(dim, heads, window, shifted=bool(index % 2)) for index in range(blocks))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.ConvTranspose2d(dim, 120, kernel_size=3, stride=3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(x).permute(0, 2, 3, 1)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).permute(0, 3, 1, 2)
        x = self.head(x)
        return x.reshape(x.shape[0], 2, 60, 81, 321)


def as_tensor(value: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.from_numpy(np.asarray(value)).to(device=device, dtype=torch.float32)


def model_input(sample: dict, device: torch.device) -> torch.Tensor:
    channels = np.concatenate((
        sample["surface"], sample["surface_mask"],
        sample["context"].reshape(-1, 81, 321),
        sample["context_mask"].reshape(-1, 81, 321),
        sample["context_offsets"], sample["context_offset_mask"],
    ), axis=0)
    return as_tensor(channels[None], device)


def sample_profiles(field: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Bilinearly sample [1,2,60,81,321] at true profile locations."""
    longitude = positions[:, 1]
    latitude = positions[:, 0]
    grid = torch.stack(((longitude - 120.0) / 160.0 * 2.0 - 1.0,
                        (latitude + 20.0) / 40.0 * 2.0 - 1.0), dim=-1)
    sampled = F.grid_sample(field.reshape(1, 120, 81, 321), grid[None, :, None, :],
                            mode="bilinear", padding_mode="border", align_corners=True)
    return sampled[0, :, :, 0].T.reshape(-1, 2, 60)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, weights=None) -> torch.Tensor:
    effective = mask if weights is None else mask * weights
    return (values * effective).sum() / effective.sum().clamp_min(1.0)


class Objective:
    def __init__(self, dataset: TrainingDataset, device: torch.device,
                 lambda_query: float, lambda_physics: float, stability_scale: float,
                 stability_tolerance: float):
        stats = dataset.statistics
        self.mean = as_tensor(np.stack((stats["target.CT_anomaly"]["mean"],
                                        stats["target.SA_anomaly"]["mean"])), device)[None, :, :, None, None]
        self.std = as_tensor(np.stack((stats["target.CT_anomaly"]["std"],
                                       stats["target.SA_anomaly"]["std"])), device)[None, :, :, None, None]
        self.lambda_query = lambda_query
        self.lambda_physics = lambda_physics
        self.stability_scale = stability_scale
        self.stability_tolerance = stability_tolerance

    def __call__(self, prediction: torch.Tensor, sample: dict, device: torch.device):
        target = as_tensor(sample["target"][None], device)
        target_mask = as_tensor(sample["target_mask"][None], device)
        area = as_tensor(sample["area_weight"][None, None, None], device)
        grid_loss = masked_mean((prediction - target).square(), target_mask, area)

        positions = as_tensor(sample["profile_position"], device)
        profile_prediction = sample_profiles(prediction, positions)
        role = torch.from_numpy(sample["profile_role_code"]).to(device)
        query_mask = as_tensor(sample["profile_mask"], device) * (role == 1)[:, None, None]
        query_target = as_tensor(sample["profile_values"], device)
        query_error = F.smooth_l1_loss(profile_prediction, query_target, reduction="none", beta=0.5)
        query_loss = masked_mean(query_error, query_mask)

        background = as_tensor(sample["background"][None], device)
        physical = background + prediction * self.std + self.mean
        ct, sa = physical[:, 0], physical[:, 1]
        alpha = as_tensor(sample["alpha"][None], device)
        beta = as_tensor(sample["beta"][None], device)
        alpha_mid = 0.5 * (alpha[:, 1:] + alpha[:, :-1])
        beta_mid = 0.5 * (beta[:, 1:] + beta[:, :-1])
        proxy = (beta_mid * (sa[:, 1:] - sa[:, :-1])
                 - alpha_mid * (ct[:, 1:] - ct[:, :-1])) / self.stability_scale
        valid = target_mask[:, :, 1:] * target_mask[:, :, :-1]
        valid = valid[:, 0] * valid[:, 1] * (alpha_mid != 0) * (beta_mid != 0)
        physics_loss = masked_mean(
            F.relu(-proxy - self.stability_tolerance).square(), valid, area[:, 0, 0])
        total = grid_loss + self.lambda_query * query_loss + self.lambda_physics * physics_loss
        return total, {"grid": grid_loss, "query": query_loss, "physics": physics_loss}

    def physical_profile_prediction(self, prediction: torch.Tensor, sample: dict,
                                    device: torch.device) -> torch.Tensor:
        normalized = sample_profiles(prediction, as_tensor(sample["profile_position"], device))
        mean = self.mean[0, :, :, 0, 0][None]
        std = self.std[0, :, :, 0, 0][None]
        background = as_tensor(sample["profile_background"], device)
        return background + normalized * std + mean


def atomic_checkpoint(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def append_json(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.no_grad()
def validate(model: nn.Module, dataset: TrainingDataset, months: list[str],
             objective: Objective, device: torch.device, amp: bool) -> dict:
    model.eval()
    normalized_sse = normalized_count = 0.0
    physical_sse = np.zeros(2, dtype="f8")
    physical_count = np.zeros(2, dtype="i8")
    for month in months:
        sample = dataset.load(month)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            prediction = model(model_input(sample, device))
        profiles = sample_profiles(prediction.float(), as_tensor(sample["profile_position"], device))
        target_normalized = as_tensor(sample["profile_values"], device)
        role = torch.from_numpy(sample["profile_role_code"]).to(device)
        mask = as_tensor(sample["profile_mask"], device).bool() & (role == 2)[:, None, None]
        normalized_sse += float(torch.where(mask, (profiles - target_normalized).square(), 0).sum())
        normalized_count += float(mask.sum())
        predicted_physical = objective.physical_profile_prediction(prediction.float(), sample, device)
        target_physical = as_tensor(sample["profile_absolute"], device)
        for variable in range(2):
            variable_mask = mask[:, variable]
            physical_sse[variable] += float(torch.where(
                variable_mask, (predicted_physical[:, variable] - target_physical[:, variable]).square(), 0).sum())
            physical_count[variable] += int(variable_mask.sum())
    return {
        "score": math.sqrt(normalized_sse / max(normalized_count, 1)),
        "CT_RMSE_C": math.sqrt(physical_sse[0] / max(physical_count[0], 1)),
        "SA_RMSE_g_kg": math.sqrt(physical_sse[1] / max(physical_count[1], 1)),
        "valid_CT_points": int(physical_count[0]),
        "valid_SA_points": int(physical_count[1]),
    }


def train(args) -> None:
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no GPU is visible")
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = TrainingDataset()
    train_months = [month for month, record in dataset.records.items() if record["split"] == "training"]
    validation_months = [month for month, record in dataset.records.items() if record["split"] == "validation"]
    if args.max_train_months:
        train_months = train_months[:args.max_train_months]
    if args.max_validation_months:
        validation_months = validation_months[:args.max_validation_months]
    probe = dataset.load(train_months[0])
    in_channels = model_input(probe, torch.device("cpu")).shape[1]
    model = SwinReconstructor(in_channels, args.dim, args.blocks, args.heads, args.window).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    amp = args.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    objective = Objective(dataset, device, args.lambda_query, args.lambda_physics,
                          args.stability_scale, args.stability_tolerance)
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    config["run_dir"] = str(config["run_dir"])
    config.update({"device_resolved": str(device), "input_channels": in_channels,
                   "train_months": train_months, "validation_months": validation_months,
                   "parameters": sum(parameter.numel() for parameter in model.parameters())})
    (args.run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    start_epoch, best_score, stale = 1, float("inf"), 0
    latest = args.run_dir / "latest.pt"
    if args.resume and latest.exists():
        checkpoint = torch.load(latest, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = checkpoint["epoch"] + 1
        best_score = checkpoint["best_score"]
        stale = checkpoint["stale"]

    print(json.dumps({"device": str(device), "parameters": config["parameters"],
                      "input_channels": in_channels, "train_months": len(train_months),
                      "validation_months": len(validation_months)}), flush=True)
    for epoch in range(start_epoch, args.epochs + 1):
        started = time.time()
        model.train()
        order = train_months.copy()
        random.Random(args.seed + epoch).shuffle(order)
        totals = {"total": 0.0, "grid": 0.0, "query": 0.0, "physics": 0.0}
        for step, month in enumerate(order, 1):
            sample = dataset.load(month)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = model(model_input(sample, device))
                loss, parts = objective(prediction, sample, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            totals["total"] += float(loss.detach())
            for name, value in parts.items():
                totals[name] += float(value.detach())
            if step % args.log_every == 0 or step == len(order):
                print(f"epoch={epoch} step={step}/{len(order)} month={month} loss={float(loss):.6f}", flush=True)
        metrics = validate(model, dataset, validation_months, objective, device, amp)
        improved = metrics["score"] < best_score
        if improved:
            best_score, stale = metrics["score"], 0
        else:
            stale += 1
        record = {"epoch": epoch, "seconds": time.time() - started,
                  **{f"train_{key}": value / len(order) for key, value in totals.items()},
                  **{f"validation_{key}": value for key, value in metrics.items()},
                  "best": improved}
        append_json(args.run_dir / "metrics.jsonl", record)
        checkpoint = {"epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scaler": scaler.state_dict(), "best_score": best_score, "stale": stale,
                      "config": config}
        atomic_checkpoint(latest, checkpoint)
        if improved:
            atomic_checkpoint(args.run_dir / "best.pt", checkpoint)
        print(json.dumps(record), flush=True)
        if stale >= args.patience:
            print(f"early_stop epoch={epoch} patience={args.patience}", flush=True)
            break


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--lambda-query", type=float, default=1.0)
    parser.add_argument("--lambda-physics", type=float, default=0.01)
    parser.add_argument("--stability-scale", type=float, default=1e-4)
    parser.add_argument("--stability-tolerance", type=float, default=0.01)
    parser.add_argument("--dim", type=int, default=48)
    parser.add_argument("--blocks", type=int, default=4)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--window", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20261117)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--cpu-threads", type=int, default=12)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--max-train-months", type=int)
    parser.add_argument("--max-validation-months", type=int)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.smoke:
        args.run_dir = args.run_dir.with_name(args.run_dir.name + "_smoke")
        args.epochs, args.patience = 1, 1
        args.dim, args.blocks, args.heads = 24, 2, 3
        args.max_train_months, args.max_validation_months = 1, 1
        args.device, args.amp, args.resume = "cpu", False, False
    return args


if __name__ == "__main__":
    train(parse_args())
