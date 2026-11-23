#!/usr/bin/env python3
"""Prepare uploaded ocean datasets for the first Swin training run.

Raw uploads remain untouched. Outputs are resumable and written below
``data/training_ready_v1`` on the server.
"""
from __future__ import annotations

import argparse
import calendar
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from itertools import repeat
from pathlib import Path

import gsw
import numpy as np
from netCDF4 import Dataset


ROOT = Path("/g3240101584lt/srtp-ocean")
RAW = ROOT / "data"
OUT = RAW / "training_ready_v1"
CODE_VERSION = "training_ready_v1_20261117"
WORKERS = 12
DEPTH = np.r_[np.arange(5, 101, 5), np.arange(110, 501, 10)].astype("f4")
SEED = "argo-v2-cqe"
EVALUATION_FRACTION = 0.2
CONTEXT_FRACTION = 0.5
SURFACE = {
    "oisst": ("sst", "sst_valid"),
    "adt": ("adt", "adt_valid"),
    "era5_u10": ("u10", "u10_valid"),
    "era5_v10": ("v10", "v10_valid"),
    "smap": ("sss_smap", "sss_smap_valid"),
}


def atomic_path(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    return temporary


def month_add(month: str, offset: int) -> str:
    year, number = int(month[:4]), int(month[4:])
    absolute = year * 12 + number - 1 + offset
    return f"{absolute // 12:04d}{absolute % 12 + 1:02d}"


def split(month: str) -> str:
    value = int(month)
    if value < 201504:
        return "archive"
    if value <= 202011:
        return "training"
    if value == 202012:
        return "isolation"
    if value <= 202212:
        return "validation"
    if value <= 202412:
        return "test_locked"
    if value <= 202506:
        return "forward_test"
    return "unassigned"


def hash_fraction(text: str) -> float:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") / 2**64


def role_for(platform: str, profile_uid: str) -> str:
    if hash_fraction(f"{SEED}:E:{platform}") < EVALUATION_FRACTION:
        return "E"
    return "C" if hash_fraction(f"{SEED}:CQ:{profile_uid}") < CONTEXT_FRACTION else "Q"


def raw_months() -> list[str]:
    return sorted(path.name.split("_")[2][:6] for path in (RAW / "cora").glob("OA_CORA5.2_*_fld_TEMP.nc"))


def selected_months(requested: list[str] | None) -> list[str]:
    months = raw_months()
    if requested:
        missing = sorted(set(requested) - set(months))
        if missing:
            raise ValueError(f"Requested months absent from CORA: {missing}")
        months = [month for month in months if month in set(requested)]
    return months


def read_array(ds: Dataset, name: str, index=None, dtype=float) -> np.ndarray:
    values = ds[name][:] if index is None else ds[name][index]
    return np.ma.filled(values, np.nan).astype(dtype)


def create_coordinate(out: Dataset, name: str, values, units: str) -> None:
    out.createDimension(name, len(values))
    variable = out.createVariable(name, "f4", (name,))
    variable[:] = values
    variable.units = units
    if name == "depth":
        variable.positive = "down"


def compressed(out: Dataset, name: str, dtype, dims, fill_value=None):
    kwargs = {"zlib": True, "complevel": 4, "shuffle": True}
    if fill_value is not None:
        kwargs["fill_value"] = fill_value
    return out.createVariable(name, dtype, dims, **kwargs)


def cora_output_ok(path: Path, month: str) -> bool:
    if not path.exists():
        return False
    try:
        with Dataset(path) as ds:
            return (getattr(ds, "processing_version", "") == CODE_VERSION
                    and getattr(ds, "month", "") == month
                    and ds["CT"].shape == (1, 60, 81, 321)
                    and set(("CT", "SA", "TEMP", "PSAL", "valid", "TEMP_ERR", "PSAL_ERR")) <= set(ds.variables))
    except OSError:
        return False


def process_cora_month(month: str, overwrite: bool = False) -> dict:
    output = OUT / "cora" / f"cora_{month}.nc"
    if not overwrite and cora_output_ok(output, month):
        return {"month": month, "output": str(output), "status": "resumed"}
    temp_path = RAW / "cora" / f"OA_CORA5.2_{month}15_fld_TEMP.nc"
    sal_path = RAW / "cora" / f"OA_CORA5.2_{month}15_fld_PSAL.nc"
    if not temp_path.exists() or not sal_path.exists():
        raise FileNotFoundError(f"{month}: missing CORA fld pair")
    with Dataset(temp_path) as temp_ds, Dataset(sal_path) as sal_ds:
        lon = read_array(temp_ds, "longitude")
        lat = read_array(temp_ds, "latitude")
        depth = read_array(temp_ds, "depth")
        ix = np.flatnonzero((np.mod(lon, 360) >= 120) & (np.mod(lon, 360) <= 280))
        iy = np.flatnonzero((lat >= -20) & (lat <= 20))
        iz = np.flatnonzero((depth >= 5) & (depth <= 500))
        ix = ix[np.argsort(np.mod(lon[ix], 360))]
        selected_depth = depth[iz].astype("f4")
        if not np.array_equal(selected_depth, DEPTH) or (len(iy), len(ix)) != (81, 321):
            raise ValueError(f"{month}: unexpected CORA target grid")
        longitude = np.mod(lon[ix], 360).astype("f4")
        latitude = lat[iy].astype("f4")
        fields = {}
        for name, ds in (("TEMP", temp_ds), ("TEMP_ERR", temp_ds), ("TEMP_PCTVAR", temp_ds),
                         ("PSAL", sal_ds), ("PSAL_ERR", sal_ds), ("PSAL_PCTVAR", sal_ds)):
            fields[name] = read_array(ds, name, (0, iz, iy, ix), dtype="f4")[None]
        temperature = fields["TEMP"][0].astype(float)
        practical_salinity = fields["PSAL"][0].astype(float)
        pressure = gsw.p_from_z(-selected_depth[:, None, None], latitude[None, :, None])
        absolute_salinity = gsw.SA_from_SP(
            practical_salinity, pressure, longitude[None, None, :], latitude[None, :, None])
        conservative_temperature = gsw.CT_from_t(absolute_salinity, temperature, pressure)
        valid = np.isfinite(absolute_salinity) & np.isfinite(conservative_temperature)
        temporary = atomic_path(output)
        with Dataset(temporary, "w", format="NETCDF4") as out:
            out.createDimension("time", 1)
            create_coordinate(out, "depth", selected_depth, "m")
            create_coordinate(out, "latitude", latitude, "degrees_north")
            create_coordinate(out, "longitude", longitude, "degrees_east")
            time = out.createVariable("time", "f8", ("time",))
            time[:] = temp_ds["time"][:]
            time.units = temp_ds["time"].units
            for name, values in fields.items():
                variable = compressed(out, name, "f4", ("time", "depth", "latitude", "longitude"), np.nan)
                variable[:] = values
                variable.units = getattr((temp_ds if name.startswith("TEMP") else sal_ds)[name], "units", "")
            for name, values, units in (
                ("CT", conservative_temperature[None], "degree_Celsius"),
                ("SA", absolute_salinity[None], "g kg-1"),
            ):
                variable = compressed(out, name, "f4", ("time", "depth", "latitude", "longitude"), np.nan)
                variable[:] = np.where(valid, values, np.nan)
                variable.units = units
            compressed(out, "valid", "u1", ("time", "depth", "latitude", "longitude"))[:] = valid[None]
            p = compressed(out, "pressure", "f4", ("depth", "latitude"), np.nan)
            p[:] = pressure[:, :, 0]
            p.units = "dbar"
            out.month = month
            out.processing_version = CODE_VERSION
            out.source_temperature = str(temp_path)
            out.source_salinity = str(sal_path)
            out.thermodynamics = f"TEOS-10 GSW {gsw.__version__}; TEMP in-situ ITS-90, PSAL PSS-78"
            out.spatial_domain = "120E-280E, 20S-20N"
        os.replace(temporary, output)
    return {"month": month, "output": str(output), "status": "processed", "valid_points": int(valid.sum())}


def load_collection() -> tuple[Path, dict]:
    path = RAW / "argo/outputs/argo_complete_v2/collection.json"
    return path, json.loads(path.read_text())


def collection_shards(month: str, collection_path: Path, collection: dict) -> list[Path]:
    return [(collection_path.parent / item["path"]).resolve() for item in collection["months"][month]]


def argo_outputs_ok(grid_path: Path, profiles_path: Path, month: str) -> bool:
    if not grid_path.exists() or not profiles_path.exists():
        return False
    try:
        with Dataset(grid_path) as grid, Dataset(profiles_path) as profiles:
            return (getattr(grid, "processing_version", "") == CODE_VERSION
                    and getattr(profiles, "processing_version", "") == CODE_VERSION
                    and getattr(grid, "month", "") == month
                    and grid["C_CT"].shape == (60, 81, 321)
                    and profiles["CT"].shape[1] == 60)
    except OSError:
        return False


def nearest_indices(grid: np.ndarray, points: np.ndarray) -> np.ndarray:
    right = np.searchsorted(grid, points)
    right = np.clip(right, 1, len(grid) - 1)
    left = right - 1
    return np.where(np.abs(points - grid[left]) <= np.abs(points - grid[right]), left, right)


def process_argo_month(month: str, longitude: np.ndarray, latitude: np.ndarray,
                       overwrite: bool = False) -> dict:
    grid_path = OUT / "argo_grid" / f"argo_grid_{month}.nc"
    profiles_path = OUT / "argo_profiles" / f"argo_profiles_{month}.nc"
    if not overwrite and argo_outputs_ok(grid_path, profiles_path, month):
        with Dataset(grid_path) as ds:
            return {"month": month, "status": "resumed", "C": int(ds.context_profiles),
                    "Q": int(ds.query_profiles), "E": int(ds.evaluation_profiles)}
    collection_path, collection = load_collection()
    arrays = {name: [] for name in ("CT", "SA", "valid", "latitude", "longitude",
                                     "observation_time", "platform_number", "profile_uid")}
    for shard in collection_shards(month, collection_path, collection):
        with Dataset(shard) as ds:
            for name in arrays:
                values = ds[name][:]
                arrays[name].append(np.asarray(values))
    arrays = {name: np.concatenate(parts) for name, parts in arrays.items()}
    arrays["platform_number"] = arrays["platform_number"].astype(str)
    arrays["profile_uid"] = arrays["profile_uid"].astype(str)
    arrays["longitude"] = np.mod(arrays["longitude"].astype(float), 360)
    arrays["latitude"] = arrays["latitude"].astype(float)
    arrays["CT"] = arrays["CT"].astype("f4")
    arrays["SA"] = arrays["SA"].astype("f4")
    arrays["valid"] = arrays["valid"].astype(bool) & np.isfinite(arrays["CT"]) & np.isfinite(arrays["SA"])
    eligible = ((arrays["latitude"] >= latitude.min()) & (arrays["latitude"] <= latitude.max())
                & (arrays["longitude"] >= longitude.min()) & (arrays["longitude"] <= longitude.max())
                & arrays["valid"].any(axis=1))
    arrays = {name: values[eligible] for name, values in arrays.items()}
    roles = np.array([role_for(platform, uid) for platform, uid in
                      zip(arrays["platform_number"], arrays["profile_uid"])])
    counts = {role: int(np.sum(roles == role)) for role in "CQE"}
    if any(value == 0 for value in counts.values()):
        raise ValueError(f"{month}: empty Argo role {counts}")

    shape = (60, len(latitude), len(longitude))
    sums_ct = np.zeros(shape, dtype="f8")
    sums_sa = np.zeros(shape, dtype="f8")
    depth_count = np.zeros(shape, dtype="u2")
    offset_sums = np.zeros((3, len(latitude), len(longitude)), dtype="f8")
    profile_count = np.zeros((len(latitude), len(longitude)), dtype="u2")
    c_index = np.flatnonzero(roles == "C")
    iy = nearest_indices(latitude, arrays["latitude"][c_index])
    ix = nearest_indices(longitude, arrays["longitude"][c_index])
    start = datetime(int(month[:4]), int(month[4:]), 1, tzinfo=timezone.utc).timestamp()
    middle_day = (calendar.monthrange(int(month[:4]), int(month[4:]))[1] + 1) / 2
    for source_index, y, x in zip(c_index, iy, ix):
        valid = arrays["valid"][source_index]
        sums_ct[valid, y, x] += arrays["CT"][source_index, valid]
        sums_sa[valid, y, x] += arrays["SA"][source_index, valid]
        depth_count[valid, y, x] += 1
        offset_sums[0, y, x] += arrays["latitude"][source_index] - latitude[y]
        offset_sums[1, y, x] += arrays["longitude"][source_index] - longitude[x]
        offset_sums[2, y, x] += (arrays["observation_time"][source_index] - start) / 86400 + 1 - middle_day
        profile_count[y, x] += 1
    c_ct = np.divide(sums_ct, depth_count, out=np.full(shape, np.nan), where=depth_count > 0)
    c_sa = np.divide(sums_sa, depth_count, out=np.full(shape, np.nan), where=depth_count > 0)
    offsets = np.divide(offset_sums, profile_count[None], out=np.full(offset_sums.shape, np.nan),
                        where=profile_count[None] > 0)
    temporary = atomic_path(grid_path)
    with Dataset(temporary, "w", format="NETCDF4") as out:
        create_coordinate(out, "depth", DEPTH, "m")
        create_coordinate(out, "latitude", latitude, "degrees_north")
        create_coordinate(out, "longitude", longitude, "degrees_east")
        for name, values, units in (("C_CT", c_ct, "degree_Celsius"), ("C_SA", c_sa, "g kg-1")):
            variable = compressed(out, name, "f4", ("depth", "latitude", "longitude"), np.nan)
            variable[:] = values
            variable.units = units
        compressed(out, "C_valid", "u1", ("depth", "latitude", "longitude"))[:] = depth_count > 0
        compressed(out, "C_count", "u2", ("depth", "latitude", "longitude"))[:] = depth_count
        for index, name, units in ((0, "C_latitude_offset", "degrees_north"),
                                   (1, "C_longitude_offset", "degrees_east"),
                                   (2, "C_time_offset", "days")):
            variable = compressed(out, name, "f4", ("latitude", "longitude"), np.nan)
            variable[:] = offsets[index]
            variable.units = units
        compressed(out, "C_profile_count", "u2", ("latitude", "longitude"))[:] = profile_count
        out.month = month
        out.processing_version = CODE_VERSION
        out.context_profiles, out.query_profiles, out.evaluation_profiles = counts["C"], counts["Q"], counts["E"]
        out.role_rule = f"E=platform SHA256 fraction {EVALUATION_FRACTION}; non-E C fraction {CONTEXT_FRACTION}; seed={SEED}"
    os.replace(temporary, grid_path)

    keep = np.flatnonzero(roles != "C")
    role_code = np.where(roles[keep] == "Q", 1, 2).astype("u1")
    temporary = atomic_path(profiles_path)
    with Dataset(temporary, "w", format="NETCDF4") as out:
        out.createDimension("profile", len(keep))
        create_coordinate(out, "depth", DEPTH, "m")
        for name, values, units in (("CT", arrays["CT"][keep], "degree_Celsius"),
                                    ("SA", arrays["SA"][keep], "g kg-1")):
            variable = compressed(out, name, "f4", ("profile", "depth"), np.nan)
            variable[:] = np.where(arrays["valid"][keep], values, np.nan)
            variable.units = units
        compressed(out, "valid", "u1", ("profile", "depth"))[:] = arrays["valid"][keep]
        role_variable = out.createVariable("role_code", "u1", ("profile",))
        role_variable[:] = role_code
        role_variable.flag_values = np.array([1, 2], dtype="u1")
        role_variable.flag_meanings = "query evaluation"
        for name in ("profile_uid", "platform_number"):
            out.createVariable(name, str, ("profile",))[:] = arrays[name][keep]
        for name, values, units in (
            ("latitude", arrays["latitude"][keep], "degrees_north"),
            ("longitude", arrays["longitude"][keep], "degrees_east"),
            ("observation_time", arrays["observation_time"][keep], "seconds since 1970-01-01 00:00:00 UTC"),
        ):
            variable = out.createVariable(name, "f8", ("profile",))
            variable[:] = values
            variable.units = units
        time_offset = (arrays["observation_time"][keep] - start) / 86400 + 1 - middle_day
        variable = out.createVariable("time_offset_days", "f4", ("profile",))
        variable[:] = time_offset
        variable.units = "days from calendar-month midpoint"
        out.month = month
        out.processing_version = CODE_VERSION
        out.query_profiles, out.evaluation_profiles = counts["Q"], counts["E"]
        out.profile_locations = "true observation positions; not snapped to grid"
    os.replace(temporary, profiles_path)
    return {"month": month, "status": "processed", **counts}


def reference_grid() -> tuple[np.ndarray, np.ndarray]:
    path = OUT / "cora" / "cora_201801.nc"
    if not path.exists():
        candidates = sorted((OUT / "cora").glob("cora_*.nc"))
        if not candidates:
            raise FileNotFoundError("No processed CORA grid")
        path = candidates[0]
    with Dataset(path) as ds:
        return read_array(ds, "longitude", dtype="f4"), read_array(ds, "latitude", dtype="f4")


def run_cora(months: list[str], overwrite: bool) -> None:
    records = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        for number, record in enumerate(executor.map(process_cora_month, months, repeat(overwrite)), 1):
            records.append(record)
            print("CORA", number, len(months), json.dumps(record), flush=True)
    path = OUT / "audits/cora_processing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n")


def run_argo(months: list[str], overwrite: bool) -> None:
    longitude, latitude = reference_grid()
    records = []
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        results = executor.map(process_argo_month, months, repeat(longitude), repeat(latitude), repeat(overwrite))
        for number, record in enumerate(results, 1):
            records.append(record)
            print("ARGO", number, len(months), json.dumps(record), flush=True)
    path = OUT / "audits/argo_processing.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2) + "\n")


def surface_path(source: str, month: str) -> Path:
    directory = source.split("_")[0]
    return RAW / directory / "months" / month / f"{directory}_{month}.nc"


def build_background(
    months: list[str],
    *,
    fitted_months: list[str] | None = None,
    destination: Path = OUT,
    version: str = CODE_VERSION,
) -> Path:
    """Fit monthly background fields without reusing held-out observations."""
    train_months = [month for month in months if split(month) == "training"] if fitted_months is None else fitted_months
    if fitted_months is None and train_months != [f"{year}{number:02d}" for year in range(2015, 2021)
                        for number in range(1, 13) if 201504 <= int(f"{year}{number:02d}") <= 202011]:
        raise ValueError("Training months are not exactly 201504-202011")
    longitude, latitude = reference_grid()
    output = destination / "statistics/training_background.nc"
    temporary = atomic_path(output)
    with Dataset(temporary, "w", format="NETCDF4") as out:
        out.createDimension("calendar_month", 12)
        month_var = out.createVariable("calendar_month", "i1", ("calendar_month",))
        month_var[:] = np.arange(1, 13)
        create_coordinate(out, "depth", DEPTH, "m")
        create_coordinate(out, "latitude", latitude, "degrees_north")
        create_coordinate(out, "longitude", longitude, "degrees_east")
        ct_out = compressed(out, "CT_background", "f4", ("calendar_month", "depth", "latitude", "longitude"), np.nan)
        sa_out = compressed(out, "SA_background", "f4", ("calendar_month", "depth", "latitude", "longitude"), np.nan)
        alpha_out = compressed(out, "alpha", "f4", ("calendar_month", "depth", "latitude", "longitude"), np.nan)
        beta_out = compressed(out, "beta", "f4", ("calendar_month", "depth", "latitude", "longitude"), np.nan)
        count_out = compressed(out, "sample_count", "u1", ("calendar_month", "depth", "latitude", "longitude"))
        pressure = gsw.p_from_z(-DEPTH[:, None, None], latitude[None, :, None])
        p_var = compressed(out, "pressure", "f4", ("depth", "latitude"), np.nan)
        p_var[:] = pressure[:, :, 0]
        p_var.units = "dbar"
        for calendar_month in range(1, 13):
            selected = [month for month in train_months if int(month[4:]) == calendar_month]
            sums_ct = np.zeros((60, 81, 321), dtype="f8")
            sums_sa = np.zeros_like(sums_ct)
            counts = np.zeros(sums_ct.shape, dtype="u1")
            for month in selected:
                with Dataset(OUT / "cora" / f"cora_{month}.nc") as ds:
                    ct, sa = read_array(ds, "CT", 0), read_array(ds, "SA", 0)
                valid = np.isfinite(ct) & np.isfinite(sa)
                sums_ct += np.where(valid, ct, 0)
                sums_sa += np.where(valid, sa, 0)
                counts += valid
            ct = np.divide(sums_ct, counts, out=np.full_like(sums_ct, np.nan), where=counts > 0)
            sa = np.divide(sums_sa, counts, out=np.full_like(sums_sa, np.nan), where=counts > 0)
            p3 = np.broadcast_to(pressure, ct.shape)
            ct_out[calendar_month - 1] = ct
            sa_out[calendar_month - 1] = sa
            alpha_out[calendar_month - 1] = gsw.alpha(sa, ct, p3)
            beta_out[calendar_month - 1] = gsw.beta(sa, ct, p3)
            count_out[calendar_month - 1] = counts
            print("BACKGROUND", calendar_month, selected, flush=True)
        out.processing_version = version
        out.fitted_months = ",".join(train_months)
        out.leakage_rule = f"Only {train_months[0]}-{train_months[-1]} training months"
        out.alpha_beta = "TEOS-10 coefficients evaluated at the training calendar-month background"
    os.replace(temporary, output)
    return output


def finish_stats(total, square, count):
    mean = np.divide(total, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0)
    variance = np.divide(square, count, out=np.full_like(total, np.nan, dtype=float), where=count > 0) - mean**2
    return {"mean": mean.tolist(), "std": np.sqrt(np.maximum(variance, 0)).tolist(), "count": count.tolist()}


def build_normalization(
    months: list[str],
    background_path: Path,
    *,
    fitted_months: list[str] | None = None,
    destination: Path = OUT,
    version: str = CODE_VERSION,
) -> Path:
    """Fit target and surface statistics on the specified training months."""
    train_months = [month for month in months if split(month) == "training"] if fitted_months is None else fitted_months
    accum = {
        "target.CT_anomaly": [np.zeros(60), np.zeros(60), np.zeros(60, dtype="i8")],
        "target.SA_anomaly": [np.zeros(60), np.zeros(60), np.zeros(60, dtype="i8")],
    }
    for source in SURFACE:
        accum[f"surface.{source}"] = [np.zeros(1), np.zeros(1), np.zeros(1, dtype="i8")]
    with Dataset(background_path) as background:
        for number, month in enumerate(train_months, 1):
            cm = int(month[4:]) - 1
            with Dataset(OUT / "cora" / f"cora_{month}.nc") as ds:
                ct, sa = read_array(ds, "CT", 0), read_array(ds, "SA", 0)
            for key, values, bg in (
                ("target.CT_anomaly", ct, read_array(background, "CT_background", cm)),
                ("target.SA_anomaly", sa, read_array(background, "SA_background", cm)),
            ):
                anomaly = values - bg
                finite = np.isfinite(anomaly)
                total, square, count = accum[key]
                total += np.where(finite, anomaly, 0).sum(axis=(1, 2))
                square += np.where(finite, anomaly * anomaly, 0).sum(axis=(1, 2))
                count += finite.sum(axis=(1, 2))
            for source, (variable, mask_name) in SURFACE.items():
                with Dataset(surface_path(source, month)) as ds:
                    values = read_array(ds, variable, 0)
                    mask = np.asarray(ds[mask_name][0]).astype(bool) & np.isfinite(values)
                total, square, count = accum[f"surface.{source}"]
                total += np.where(mask, values, 0).sum()
                square += np.where(mask, values * values, 0).sum()
                count += mask.sum()
            print("NORMALIZATION", number, len(train_months), month, flush=True)
    output = destination / "statistics/normalization.json"
    output.write_text(json.dumps({
        "processing_version": version,
        "fitted_months": train_months,
        "statistics": {key: finish_stats(*values) for key, values in accum.items()},
    }, indent=2) + "\n")
    return output


def run_statistics(months: list[str]) -> None:
    background = build_background(months)
    normalization = build_normalization(months, background)
    print("STATISTICS", background, normalization, flush=True)


def build_manifest(months: list[str]) -> Path:
    records = []
    for month in months:
        history = {}
        for source, (variable, mask) in SURFACE.items():
            history[source] = [{
                "month": history_month,
                "path": str(surface_path(source, history_month)),
                "variable": variable,
                "valid_mask": mask,
            } for history_month in (month_add(month, -2), month_add(month, -1), month)]
            if any(not Path(item["path"]).exists() for item in history[source]):
                raise FileNotFoundError(f"{month}: missing {source} history")
        records.append({
            "month": month,
            "split": split(month),
            "cora": str(OUT / "cora" / f"cora_{month}.nc"),
            "argo_grid": str(OUT / "argo_grid" / f"argo_grid_{month}.nc"),
            "argo_profiles": str(OUT / "argo_profiles" / f"argo_profiles_{month}.nc"),
            "surface_history_t_minus_2_to_t": history,
            "background": str(OUT / "statistics/training_background.nc"),
            "calendar_month_index": int(month[4:]) - 1,
        })
    output = OUT / "training_manifest.json"
    output.write_text(json.dumps({
        "schema": "swin_training_manifest_v1",
        "processing_version": CODE_VERSION,
        "domain": {"longitude": [120, 280], "latitude": [-20, 20], "depth_m": [5, 500],
                   "shape": [60, 81, 321]},
        "role_rule": {"seed": SEED, "evaluation_platform_fraction": EVALUATION_FRACTION,
                      "context_profile_fraction_of_non_evaluation": CONTEXT_FRACTION},
        "surface_valid_rule": "provided valid flag equals 1 and value is finite",
        "normalization": str(OUT / "statistics/normalization.json"),
        "months": records,
    }, indent=2) + "\n")
    print("MANIFEST", output, len(records), flush=True)
    return output


def validate(months: list[str]) -> Path:
    manifest_path = OUT / "training_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    records = manifest["months"]
    if [record["month"] for record in records] != months:
        raise ValueError("Manifest months differ from CORA months")
    normalization = json.loads((OUT / "statistics/normalization.json").read_text())
    expected_train = [month for month in months if split(month) == "training"]
    if normalization["fitted_months"] != expected_train:
        raise ValueError("Normalization leakage or missing training month")
    longitude, latitude = reference_grid()
    seen_uids = set()
    e_platforms, development_platforms = set(), set()
    role_totals = {"C": 0, "Q": 0, "E": 0}
    checks = []
    for number, record in enumerate(records, 1):
        month = record["month"]
        with Dataset(record["cora"]) as cora:
            if not np.array_equal(read_array(cora, "longitude", dtype="f4"), longitude):
                raise ValueError(f"{month}: CORA longitude mismatch")
            valid = np.asarray(cora["valid"][:]).astype(bool)
            ct, sa = read_array(cora, "CT"), read_array(cora, "SA")
            if not np.array_equal(valid, np.isfinite(ct) & np.isfinite(sa)) or not valid.any():
                raise ValueError(f"{month}: invalid CORA mask")
        with Dataset(record["argo_grid"]) as grid:
            c_valid = np.asarray(grid["C_valid"][:]).astype(bool)
            if not c_valid.any() or not np.isfinite(read_array(grid, "C_CT"))[c_valid].all():
                raise ValueError(f"{month}: invalid C grid")
            month_counts = {role: int(getattr(grid, {"C": "context_profiles", "Q": "query_profiles",
                                                       "E": "evaluation_profiles"}[role])) for role in "CQE"}
        with Dataset(record["argo_profiles"]) as profiles:
            roles = np.asarray(profiles["role_code"][:])
            uids = np.asarray(profiles["profile_uid"][:]).astype(str)
            platforms = np.asarray(profiles["platform_number"][:]).astype(str)
            values_valid = np.asarray(profiles["valid"][:]).astype(bool)
            if not set(np.unique(roles)).issubset({1, 2}) or not values_valid.any():
                raise ValueError(f"{month}: invalid Q/E profile file")
            for uid in uids:
                if uid in seen_uids:
                    raise ValueError(f"Duplicate profile UID: {uid}")
                seen_uids.add(uid)
            for role, platform in zip(roles, platforms):
                expected_e = hash_fraction(f"{SEED}:E:{platform}") < EVALUATION_FRACTION
                if (role == 2) != expected_e:
                    raise ValueError(f"{month}: platform role inconsistency")
                (e_platforms if role == 2 else development_platforms).add(platform)
            if month_counts["Q"] != int((roles == 1).sum()) or month_counts["E"] != int((roles == 2).sum()):
                raise ValueError(f"{month}: role count mismatch")
        for source, items in record["surface_history_t_minus_2_to_t"].items():
            variable, mask_name = SURFACE[source]
            for item in items:
                with Dataset(item["path"]) as ds:
                    if not np.array_equal(read_array(ds, "longitude"), longitude.astype(float)):
                        raise ValueError(f"{month}: {source} grid mismatch")
                    values = read_array(ds, variable, 0)
                    mask = np.asarray(ds[mask_name][0]).astype(bool)
                    if np.any(mask & ~np.isfinite(values)):
                        raise ValueError(f"{month}: {source} valid mask includes nonfinite value")
        for role in "CQE":
            role_totals[role] += month_counts[role]
        checks.append({"month": month, "split": record["split"], **month_counts,
                       "cora_valid_fraction": float(valid.mean()),
                       "C_grid_valid_points": int(c_valid.sum())})
        print("VALIDATE", number, len(records), month, flush=True)
    if e_platforms & development_platforms:
        raise ValueError("An E platform appears in Q")
    if any(value == 0 for value in role_totals.values()):
        raise ValueError("Empty aggregate Argo role")
    report = {
        "status": "passed",
        "processing_version": CODE_VERSION,
        "months": len(records),
        "first_month": months[0],
        "last_month": months[-1],
        "training_months": len(expected_train),
        "role_totals": role_totals,
        "unique_QE_profile_uids": len(seen_uids),
        "evaluation_platforms": len(e_platforms),
        "development_platforms_with_Q": len(development_platforms),
        "checks": checks,
    }
    output = OUT / "audits/training_readiness.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print("VALIDATION_PASSED", json.dumps({key: report[key] for key in
          ("months", "training_months", "role_totals", "evaluation_platforms")}), flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", choices=("cora", "argo", "statistics", "manifest", "validate"),
                        default=["cora", "argo", "statistics", "manifest", "validate"])
    parser.add_argument("--months", nargs="+", help="Optional YYYYMM subset for CORA/Argo pilot processing")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    months = selected_months(args.months)
    OUT.mkdir(parents=True, exist_ok=True)
    if "cora" in args.stages:
        run_cora(months, args.overwrite)
    if "argo" in args.stages:
        run_argo(months, args.overwrite)
    if args.months and any(stage in args.stages for stage in ("statistics", "manifest", "validate")):
        raise ValueError("Subset months are only allowed for cora/argo pilot stages")
    all_months = raw_months()
    if "statistics" in args.stages:
        run_statistics(all_months)
    if "manifest" in args.stages:
        build_manifest(all_months)
    if "validate" in args.stages:
        validate(all_months)


if __name__ == "__main__":
    main()
