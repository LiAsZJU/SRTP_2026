#!/usr/bin/env python3
"""Fetch RSS V6 December source windows ending before the next split."""

from __future__ import annotations

import argparse
import concurrent.futures
import time
from datetime import UTC, date, datetime
from pathlib import Path

import aiohttp
import fsspec
import h5py
import numpy as np

BASE = "https://data.remss.com/smap/SSS/V06.0/FINAL/L3/8day_running"


def coverage_date(dataset: h5py.File, key: str) -> datetime:
    """Decode the provider's UTC coverage time from HDF5 attributes."""
    value = dataset.attrs[key]
    if isinstance(value, np.ndarray):
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("ascii")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def fetch_window(job: tuple[int, int, Path]) -> str:
    """Save the selected source-grid salinity and quality-screened mask."""
    year, day, directory = job
    nominal = date(year, 12, day)
    ordinal = nominal.timetuple().tm_yday
    name = f"RSS_smap_SSS_L3_8day_running_{year}_{ordinal:03d}_FNL_v06.0.nc"
    url = f"{BASE}/{year}/{name}"
    output = directory / f"{nominal:%Y%m%d}.npz"
    if output.exists():
        return output.name
    for attempt in range(3):
        try:
            with fsspec.open(
                url,
                "rb",
                block_size=1 << 20,
                cache_type="blockcache",
                client_kwargs={"timeout": aiohttp.ClientTimeout(total=70, sock_connect=12, sock_read=30)},
            ) as stream, h5py.File(stream, "r") as source:
                start = coverage_date(source, "time_coverage_start")
                end = coverage_date(source, "time_coverage_end")
                if end > datetime(year + 1, 1, 1, tzinfo=UTC):
                    raise ValueError(f"{name} exceeds split cutoff: {end}")
                if (end - start).total_seconds() != 8 * 86400:
                    raise ValueError(f"{name} does not contain an 8-day window")
                lat_ref = source["lat"][:]
                lon_ref = source["lon"][:]
                lat_index = np.flatnonzero((lat_ref >= -21) & (lat_ref <= 21))
                lon_index = np.flatnonzero((lon_ref >= 119) & (lon_ref <= 281))
                ys = slice(int(lat_index[0]), int(lat_index[-1]) + 1)
                xs = slice(int(lon_index[0]), int(lon_index[-1]) + 1)
                raw = source["sss_smap"][ys, xs]
                observations = source["nobs"][ys, xs]
                field = source["sss_smap"]
                scale = float(np.asarray(field.attrs["scale_factor"]).item())
                offset = float(np.asarray(field.attrs["add_offset"]).item())
                low = float(np.asarray(field.attrs["valid_min"]).item())
                high = float(np.asarray(field.attrs["valid_max"]).item())
                missing = float(np.asarray(field.attrs["_FillValue"]).item())
                valid = np.isfinite(raw) & (raw != missing) & (raw >= low) & (raw <= high) & (observations >= 1)
                salinity = np.where(valid, raw * scale + offset, np.nan).astype("f4")
            with output.with_suffix(".partial").open("wb") as stream:
                np.savez_compressed(
                    stream,
                    latitude=lat_ref[ys],
                    longitude=lon_ref[xs],
                    salinity=salinity,
                    source_url=url,
                    nominal_date=nominal.isoformat(),
                    support_start=start.isoformat(),
                    support_end=end.isoformat(),
                )
            output.with_suffix(".partial").replace(output)
            return output.name
        except (OSError, TimeoutError, aiohttp.ClientError):
            if attempt == 2:
                raise
            time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{name} did not download")


def main() -> None:
    """Fetch the 27 eligible December windows for each test boundary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.directory.mkdir(parents=True, exist_ok=True)
    jobs = [(year, day, args.directory) for year in (2023, 2024) for day in range(1, 28)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch_window, job) for job in jobs]
        for number, future in enumerate(concurrent.futures.as_completed(futures), 1):
            print(f"{number}/{len(jobs)} {future.result()}", flush=True)


if __name__ == "__main__":
    main()
