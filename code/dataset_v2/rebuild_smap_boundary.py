#!/usr/bin/env python3
"""Rebuild boundary SMAP monthly inputs from eligible original RSS windows."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset

from build_dataset import DATASET, write_json
from prepare_server_training_data import RAW

MONTHS = ("202312", "202412")


def overlap_weights(target: np.ndarray, centres: np.ndarray, *, latitude: bool) -> np.ndarray:
    """Return separable spherical overlap factors for native quarter-degree cells."""
    lower = np.maximum(target[:, 0, None], centres[None, :] - 0.125)
    upper = np.minimum(target[:, 1, None], centres[None, :] + 0.125)
    if latitude:
        return np.maximum(np.sin(np.deg2rad(upper)) - np.sin(np.deg2rad(lower)), 0.0)
    return np.maximum(np.deg2rad(upper - lower), 0.0)


def safe_divide(numerator: np.ndarray, divisor: np.ndarray) -> np.ndarray:
    """Divide on valid support and otherwise return NaN."""
    return np.divide(numerator, divisor, out=np.full(divisor.shape, np.nan), where=divisor > 0)


def support_envelope(
    earliest: np.ndarray,
    latest: np.ndarray,
    y_weights: np.ndarray,
    x_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map source support bounds to target cells touched by valid source support."""
    start = np.empty((y_weights.shape[0], x_weights.shape[0]))
    end = np.empty_like(start)
    for row, y in enumerate(y_weights > 0):
        for column, x in enumerate(x_weights > 0):
            start[row, column] = earliest[np.ix_(y, x)].min(initial=np.inf)
            end[row, column] = latest[np.ix_(y, x)].max(initial=-np.inf)
    return np.where(np.isfinite(start), start, np.nan), np.where(np.isfinite(end), end, np.nan)


def rebuild_month(month: str, source_directory: Path) -> dict:
    """Rebuild one month with archived QC, geometry, and a strict support cutoff."""
    original_root = RAW / "smap/months" / month
    original = original_root / f"smap_{month}.nc"
    inventory = json.loads((original_root / "source_inventory.json").read_text(encoding="utf-8"))
    cutoff = f"{int(month[:4]) + 1}-01-01T00:00:00+00:00"
    cutoff_dt = datetime.fromisoformat(cutoff)
    eligible = {}
    excluded = []
    for window in inventory:
        name = Path(window["path"]).name
        metadata = window["global_attributes"]
        start = datetime.fromisoformat(metadata["time_coverage_start"])
        end = datetime.fromisoformat(metadata["time_coverage_end"])
        if end > cutoff_dt:
            excluded.append({"granule": name, "support_start": start.isoformat(), "support_end": end.isoformat()})
        else:
            eligible[name] = (start, end)
    paths = sorted(source_directory.glob(f"{month}??.npz"))
    expected_names = set(eligible)
    if len(paths) != len(eligible) or len(paths) != 27:
        raise ValueError(f"{month}: expected 27 eligible windows, received {len(paths)}")
    used = []
    lat = lon = None
    for path in paths:
        with np.load(path, allow_pickle=False) as window:
            url = str(window["source_url"].item())
            name = Path(url).name
            start = datetime.fromisoformat(str(window["support_start"].item()))
            end = datetime.fromisoformat(str(window["support_end"].item()))
            if name not in expected_names or eligible[name] != (start, end):
                raise ValueError(f"{month}: unexpected granule or mismatched support: {name}")
            nominal = str(window["nominal_date"].item())
            if nominal.replace("-", "") != path.stem:
                raise ValueError(f"{month}: wrong nominal date in {path.name}")
            window_lat = window["latitude"].astype("f8")
            window_lon = window["longitude"].astype("f8")
            y = (window_lat >= -20) & (window_lat <= 20)
            x = (window_lon >= 120) & (window_lon <= 280)
            if lat is None:
                lat, lon = window_lat[y], window_lon[x]
                if not np.allclose(np.diff(lat), 0.25) or not np.allclose(np.diff(lon), 0.25):
                    raise ValueError("Unexpected native source grid")
                shape = (len(lat), len(lon))
                totals = np.zeros(shape, dtype="f8")
                counts = np.zeros(shape, dtype="i4")
                earliest = np.full(shape, np.inf)
                latest = np.full(shape, -np.inf)
            elif not np.array_equal(lat, window_lat[y]) or not np.array_equal(lon, window_lon[x]):
                raise ValueError(f"Source grid drift in {name}")
            values = window["salinity"][np.ix_(y, x)]
            valid = np.isfinite(values)
            totals[valid] += values[valid]
            counts[valid] += 1
            earliest[valid] = np.minimum(earliest[valid], start.timestamp())
            latest[valid] = np.maximum(latest[valid], end.timestamp())
            expected_names.remove(name)
            used.append({
                "granule": name,
                "source_url": url,
                "nominal_date": nominal,
                "support_start": start.isoformat(),
                "support_end": end.isoformat(),
                "source_subset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            })
    if expected_names or lat is None or lon is None:
        raise ValueError(f"{month}: missing eligible windows {expected_names}")
    destination = DATASET / "smap_boundary" / original.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        raise FileExistsError(destination)
    temporary = destination.with_suffix(".nc.tmp")
    shutil.copyfile(original, temporary)
    with Dataset(temporary, "r+") as out:
        area = np.asarray(out["cell_area"][:])
        radius_squared = float(out.earth_radius_m) ** 2
        y_weights = overlap_weights(np.asarray(out["latitude_bounds"][:]), lat, latitude=True)
        x_weights = overlap_weights(np.asarray(out["longitude_bounds"][:]), lon, latitude=False)
        geometric_area = radius_squared * np.outer(y_weights.sum(axis=1), x_weights.sum(axis=1))
        geom = geometric_area / area
        if not np.allclose(geom, out["smap_geometry_coverage"][0], rtol=0, atol=1e-9):
            raise ValueError(f"{month}: source crop does not reproduce archived geometry")
        geom_complete = np.abs(geom - 1) <= float(out.geometry_tolerance)
        valid_source = counts > 0
        valid_area = radius_squared * (y_weights @ valid_source.astype(float) @ x_weights.T)
        source_means = np.divide(totals, counts, out=np.zeros_like(totals), where=valid_source)
        field = safe_divide(radius_squared * (y_weights @ source_means @ x_weights.T), valid_area)
        valid_target = geom_complete & np.isfinite(field)
        count_field = safe_divide(radius_squared * (y_weights @ counts @ x_weights.T), valid_area)
        start_field, end_field = support_envelope(earliest, latest, y_weights, x_weights)
        if np.nanmax(end_field) > cutoff_dt.timestamp() or not valid_target.any():
            raise ValueError(f"{month}: invalid target time support")
        fields = {
            "smap_geometry_coverage": geom,
            "sss_smap": np.where(valid_target, field, np.nan),
            "sss_smap_valid": valid_target.astype("i4"),
            "sss_smap_valid_area_m2": valid_area,
            "sss_smap_valid_area_fraction": valid_area / area,
            "sss_smap_valid_fraction_of_covered_area": safe_divide(valid_area, geometric_area),
            "sss_smap_geometry_complete": geom_complete.astype("i4"),
            "sss_smap_valid_windows_area_mean": count_field,
            "sss_smap_window_support_start": start_field,
            "sss_smap_window_support_end": end_field,
        }
        for name, values in fields.items():
            out[name][0] = values
        out.window_support_start = min(item["support_start"] for item in used)
        out.window_support_end = max(item["support_end"] for item in used)
        out.split_window_policy = f"Whole source windows ending at or before {cutoff}; spatial and QC rules preserved"
        out.source_window_count = len(used)
        out.excluded_source_window_count = len(excluded)
        out.boundary_source_inventory = f"audits/smap_boundary_{month}.json"
    temporary.replace(destination)
    result = {
        "month": month,
        "cutoff_utc": cutoff,
        "output": str(destination),
        "valid_target_cells": int(valid_target.sum()),
        "used_windows": used,
        "excluded_windows": excluded,
        "window_support_end": max(item["support_end"] for item in used),
    }
    print("SMAP_REBUILT", month, len(used), len(excluded), result["window_support_end"], result["valid_target_cells"])
    return result


def main() -> None:
    """Rebuild both cross-split months and publish provenance records."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("window_directory", type=Path)
    parser.add_argument("--months", choices=MONTHS, nargs="+", default=MONTHS)
    args = parser.parse_args()
    for month in args.months:
        result = rebuild_month(month, args.window_directory)
        write_json(DATASET / "audits" / f"smap_boundary_{month}.json", result)


if __name__ == "__main__":
    main()
