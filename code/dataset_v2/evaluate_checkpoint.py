#!/usr/bin/env python3
"""Score a frozen checkpoint on held-out profiles and platform cohorts."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import numpy as np
import torch
from netCDF4 import Dataset

from build_dataset import write_json
from dataset_v2.grid_sampler import GridSampler
from server_training_data import ROOT as BASELINE_DATASET
from server_training_data import TrainingDataset
from train_swin_physics import Objective, SwinReconstructor, as_tensor, model_input, sample_profiles


class Metrics:
    """Accumulate pooled physical moments and normalized profile error."""

    def __init__(self) -> None:
        """Initialize moments for CT and SA."""
        self.values = np.zeros((2, 6), dtype="f8")
        self.profiles = 0
        self.platforms: set[str] = set()

    def add(
        self,
        target: np.ndarray,
        predicted: np.ndarray,
        normalized_error: np.ndarray,
        valid: np.ndarray,
        cohort: np.ndarray,
        platform: np.ndarray,
    ) -> None:
        """Accumulate valid Argo levels and identify contributing platforms."""
        good_profiles = cohort & valid.any(axis=(1, 2))
        self.profiles += int(good_profiles.sum())
        self.platforms.update(platform[good_profiles].tolist())
        for variable in range(2):
            mask = cohort[:, None] & valid[:, variable]
            observed = target[:, variable][mask].astype("f8")
            error = predicted[:, variable][mask].astype("f8") - observed
            normalized = normalized_error[:, variable][mask].astype("f8")
            self.values[variable] += (
                observed.size,
                observed.sum(),
                np.square(observed).sum(),
                error.sum(),
                np.square(error).sum(),
                np.square(normalized).sum(),
            )

    def summary(self) -> dict:
        """Return scores pooled over valid months, profiles, and depths."""
        output = {"profiles_with_valid_levels": self.profiles, "platforms": len(self.platforms)}
        norm_count = float(self.values[:, 0].sum())
        output["normalized_score"] = math.sqrt(self.values[:, 5].sum() / norm_count) if norm_count else None
        for variable, name in enumerate(("CT", "SA")):
            count, total, sum_squares, error_sum, sse, _ = self.values[variable]
            if count:
                sst = sum_squares - total * total / count
                output[name] = {
                    "valid_points": int(count),
                    "rmse": math.sqrt(sse / count),
                    "bias": error_sum / count,
                    "r2": 1 - sse / sst if sst > 0 else None,
                }
            else:
                output[name] = {"valid_points": 0, "rmse": None, "bias": None, "r2": None}
        return output


def load_profiles(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Read role codes and platform identifiers aligned with physical values."""
    with Dataset(path) as dataset:
        return np.asarray(dataset["role_code"][:]), np.asarray(dataset["platform_number"][:], dtype=str)


def eligible_validation_platforms(source: TrainingDataset) -> set[str]:
    """Return the platforms used for checkpoint selection in this protocol."""
    platforms = set()
    for row in source.records.values():
        if row["split"] == "validation":
            role, label = load_profiles(row["argo_profiles"])
            platforms.update(label[role == 2].tolist())
    return platforms


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    """Score one immutable checkpoint on the dataset it was trained for."""
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(args.cpu_threads)
    device = torch.device(args.device)
    source = TrainingDataset(args.dataset_dir)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    config = checkpoint["config"]
    manifest_hash = hashlib.sha256((args.dataset_dir / "training_manifest.json").read_bytes()).hexdigest()
    expected_hash = config.get("dataset_manifest_sha256")
    if expected_hash is None:
        if args.dataset_dir.resolve() != BASELINE_DATASET.resolve():
            raise ValueError("Legacy checkpoint requires its original dataset and normalization")
    elif expected_hash != manifest_hash:
        raise ValueError("Checkpoint and dataset manifest do not match")
    sampler = GridSampler(Path(next(iter(source.records.values()))["background"]), device)
    estimators = {"native_coordinates": sampler}
    if args.compare_legacy:
        estimators["legacy_uniform_latitude"] = sample_profiles
    model = SwinReconstructor(
        config["input_channels"], config["dim"], config["blocks"], config["heads"], config["window"]
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    objective = Objective(
        source, device, config["lambda_query"], config["lambda_physics"],
        config["stability_scale"], config["stability_tolerance"],
    )
    mean = objective.mean[0, :, :, 0, 0]
    std = objective.std[0, :, :, 0, 0]
    validation_platforms = sorted(eligible_validation_platforms(source))
    report = {
        "checkpoint": str(args.checkpoint.resolve()),
        "epoch": checkpoint["epoch"],
        "dataset": str(args.dataset_dir.resolve()),
        "manifest_sha256": manifest_hash,
        "device": str(device),
        "metrics": {},
    }
    for split in args.splits:
        months = [month for month, row in source.records.items() if row["split"] == split]
        if not months:
            raise ValueError(f"Empty or unknown split {split}")
        cohort_names = ["all_E"]
        if split != "validation":
            cohort_names.append("E_platforms_unseen_in_validation")
        metrics = {name: {cohort: Metrics() for cohort in cohort_names} for name in estimators}
        for month in months:
            sample = source.load(month)
            role, platform = load_profiles(source.records[month]["argo_profiles"])
            cohort_masks = {
                "all_E": role == 2,
                "E_platforms_unseen_in_validation": (role == 2) & ~np.isin(platform, validation_platforms),
            }
            valid = sample["profile_mask"].astype(bool)
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda" and config["amp"]
            ):
                predicted_grid = model(model_input(sample, device))
            positions = as_tensor(sample["profile_position"], device)
            background = as_tensor(sample["profile_background"], device)
            for name, operator in estimators.items():
                normalized = operator(predicted_grid.float(), positions)
                physical = background + normalized * std + mean
                errors = normalized.cpu().numpy() - sample["profile_values"]
                physical = physical.cpu().numpy()
                for cohort, accum in metrics[name].items():
                    accum.add(sample["profile_absolute"], physical, errors, valid, cohort_masks[cohort], platform)
        report["metrics"][split] = {
            "months": months,
            "operators": {name: {cohort: item.summary() for cohort, item in values.items()}
                          for name, values in metrics.items()},
        }
        print(split, {name: values["all_E"].summary() for name, values in metrics.items()}, flush=True)
    return report


def main() -> None:
    """Evaluate the requested splits and write a versioned JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--splits", nargs="+", choices=("validation", "test_locked", "forward_test"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--compare-legacy", action="store_true")
    args = parser.parse_args()
    write_json(args.output, evaluate(args))


if __name__ == "__main__":
    main()
