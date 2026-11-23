#!/usr/bin/env python3
"""Load one training-ready month as finite NumPy tensors plus explicit masks."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from netCDF4 import Dataset


ROOT = Path("/g3240101584lt/srtp-ocean/data/training_ready_v1")


def read(ds: Dataset, name: str, index=None, dtype="f4") -> np.ndarray:
    values = ds[name][:] if index is None else ds[name][index]
    return np.ma.filled(values, np.nan).astype(dtype)


def normalize(values: np.ndarray, mean, std, depth_axis: int | None = None) -> np.ndarray:
    mean = np.asarray(mean, dtype="f4")
    std = np.asarray(std, dtype="f4")
    if depth_axis is not None:
        shape = [1] * values.ndim
        shape[depth_axis] = len(mean)
        mean, std = mean.reshape(shape), std.reshape(shape)
    return (values - mean) / np.where(std > 0, std, 1)


def horizontal_sample(field: np.ndarray, grid_lat: np.ndarray, grid_lon: np.ndarray,
                      points_lat: np.ndarray, points_lon: np.ndarray) -> np.ndarray:
    """Mask-aware bilinear sampling; output shape is [depth, profile]."""
    y1 = np.clip(np.searchsorted(grid_lat, points_lat), 1, len(grid_lat) - 1)
    x1 = np.clip(np.searchsorted(grid_lon, points_lon), 1, len(grid_lon) - 1)
    y0, x0 = y1 - 1, x1 - 1
    fy = (points_lat - grid_lat[y0]) / (grid_lat[y1] - grid_lat[y0])
    fx = (points_lon - grid_lon[x0]) / (grid_lon[x1] - grid_lon[x0])
    corners = ((y0, x0, (1 - fy) * (1 - fx)), (y0, x1, (1 - fy) * fx),
               (y1, x0, fy * (1 - fx)), (y1, x1, fy * fx))
    total = np.zeros((field.shape[0], len(points_lat)), dtype="f8")
    support = np.zeros_like(total)
    for iy, ix, weight in corners:
        values = field[:, iy, ix]
        finite = np.isfinite(values)
        total += np.where(finite, values * weight[None], 0)
        support += finite * weight[None]
    return np.divide(total, support, out=np.full_like(total, np.nan), where=support > 0)


class TrainingDataset:
    def __init__(self, root: Path = ROOT):
        self.root = root
        manifest = json.loads((root / "training_manifest.json").read_text())
        self.records = {record["month"]: record for record in manifest["months"]}
        self.statistics = json.loads(Path(manifest["normalization"]).read_text())["statistics"]

    def load(self, month: str) -> dict[str, np.ndarray | str]:
        record = self.records[month]
        calendar_month = record["calendar_month_index"]
        with Dataset(record["background"]) as ds:
            background_ct = read(ds, "CT_background", calendar_month)
            background_sa = read(ds, "SA_background", calendar_month)
            alpha = read(ds, "alpha", calendar_month)
            beta = read(ds, "beta", calendar_month)
            latitude, longitude = read(ds, "latitude"), read(ds, "longitude")
        backgrounds = np.stack((background_ct, background_sa))
        background_mask = np.isfinite(backgrounds)

        surface_values, surface_masks = [], []
        for source, history in record["surface_history_t_minus_2_to_t"].items():
            stats = self.statistics[f"surface.{source}"]
            for item in history:
                with Dataset(item["path"]) as ds:
                    values = read(ds, item["variable"], 0)
                    mask = np.asarray(ds[item["valid_mask"]][0]).astype(bool) & np.isfinite(values)
                scaled = normalize(values, stats["mean"], stats["std"])
                surface_values.append(np.where(mask, scaled, 0))
                surface_masks.append(mask.astype("f4"))

        target_stats = [self.statistics["target.CT_anomaly"], self.statistics["target.SA_anomaly"]]
        with Dataset(record["cora"]) as ds:
            target_absolute = np.stack((read(ds, "CT", 0), read(ds, "SA", 0)))
            target_mask = np.asarray(ds["valid"][0]).astype(bool)[None].repeat(2, axis=0)
        target_anomaly = target_absolute - backgrounds
        target = np.stack([normalize(target_anomaly[i], target_stats[i]["mean"],
                                     target_stats[i]["std"], depth_axis=0) for i in range(2)])
        target_mask &= np.isfinite(target)
        target = np.where(target_mask, target, 0).astype("f4")

        with Dataset(record["argo_grid"]) as ds:
            context_absolute = np.stack((read(ds, "C_CT"), read(ds, "C_SA")))
            context_mask = np.asarray(ds["C_valid"][:]).astype(bool)[None].repeat(2, axis=0)
            offsets = np.stack((read(ds, "C_latitude_offset"), read(ds, "C_longitude_offset"),
                                read(ds, "C_time_offset")))
        context_anomaly = context_absolute - backgrounds
        context = np.stack([normalize(context_anomaly[i], target_stats[i]["mean"],
                                      target_stats[i]["std"], depth_axis=0) for i in range(2)])
        context_mask &= np.isfinite(context)
        context = np.where(context_mask, context, 0).astype("f4")
        offset_mask = np.isfinite(offsets)
        offsets = np.where(offset_mask, offsets, 0).astype("f4")

        with Dataset(record["argo_profiles"]) as ds:
            role_code = np.asarray(ds["role_code"][:])
            profile_absolute = np.stack((read(ds, "CT"), read(ds, "SA")), axis=1)
            profile_mask = np.asarray(ds["valid"][:]).astype(bool)[:, None].repeat(2, axis=1)
            profile_lat, profile_lon = read(ds, "latitude"), read(ds, "longitude")
            profile_time = read(ds, "time_offset_days")
        sampled_background = np.stack((
            horizontal_sample(background_ct, latitude, longitude, profile_lat, profile_lon).T,
            horizontal_sample(background_sa, latitude, longitude, profile_lat, profile_lon).T,
        ), axis=1)
        profile_anomaly = profile_absolute - sampled_background
        profiles = np.stack([normalize(profile_anomaly[:, i], target_stats[i]["mean"],
                                       target_stats[i]["std"], depth_axis=1) for i in range(2)], axis=1)
        profile_mask &= np.isfinite(profiles)
        profiles = np.where(profile_mask, profiles, 0).astype("f4")

        result: dict[str, np.ndarray | str] = {
            "month": month,
            "split": record["split"],
            "surface": np.stack(surface_values).astype("f4"),
            "surface_mask": np.stack(surface_masks),
            "context": context,
            "context_mask": context_mask.astype("f4"),
            "context_offsets": offsets,
            "context_offset_mask": offset_mask.astype("f4"),
            "target": target,
            "target_mask": target_mask.astype("f4"),
            "target_absolute": np.where(target_mask, target_absolute, 0).astype("f4"),
            "background": np.where(background_mask, backgrounds, 0).astype("f4"),
            "background_mask": background_mask.astype("f4"),
            "alpha": np.where(np.isfinite(alpha), alpha, 0).astype("f4"),
            "beta": np.where(np.isfinite(beta), beta, 0).astype("f4"),
            "profile_values": profiles,
            "profile_mask": profile_mask.astype("f4"),
            "profile_absolute": np.where(profile_mask, profile_absolute, 0).astype("f4"),
            "profile_background": np.where(profile_mask, sampled_background, 0).astype("f4"),
            "profile_role_code": role_code.astype("u1"),
            "profile_position": np.stack((profile_lat, profile_lon, profile_time), axis=1).astype("f4"),
            "area_weight": np.broadcast_to(np.cos(np.deg2rad(latitude))[:, None],
                                            (len(latitude), len(longitude))).astype("f4"),
        }
        for key, value in result.items():
            if isinstance(value, np.ndarray) and not np.isfinite(value).all():
                raise ValueError(f"{month}: nonfinite returned tensor {key}")
        return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("months", nargs="+", default=["201504"])
    args = parser.parse_args()
    dataset = TrainingDataset()
    for month in args.months:
        sample = dataset.load(month)
        shapes = {key: list(value.shape) for key, value in sample.items() if isinstance(value, np.ndarray)}
        mask_sums = {key: int(value.sum()) for key, value in sample.items()
                     if isinstance(value, np.ndarray) and key.endswith("mask")}
        roles, counts = np.unique(sample["profile_role_code"], return_counts=True)
        print(json.dumps({"month": month, "split": sample["split"], "shapes": shapes,
                          "mask_sums": mask_sums,
                          "profile_roles": {str(role): int(count) for role, count in zip(roles, counts)}},
                         indent=2))


if __name__ == "__main__":
    main()
