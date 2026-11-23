#!/usr/bin/env python3
"""Audit held-out Argo matches in the CORA analysis observation inventories."""

from __future__ import annotations

from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, chartostring

from build_dataset import DATASET, baseline_manifest, split_for, write_json
from prepare_server_training_data import RAW

TIME_TOLERANCE_SECONDS = 3600
POSITION_TOLERANCE_DEGREES = 0.01
CHECKED_SPLITS = ("training", "validation", "test_locked", "forward_test")


def reference_timestamp(dataset: Dataset) -> float:
    """Read the reference UTC epoch of a CORA JULD vector."""
    raw = np.ma.filled(dataset["REFERENCE_DATE_TIME"][:], b"").tobytes().decode("ascii")
    digits = "".join(character for character in raw if character.isdigit())
    return datetime.strptime(digits[:14], "%Y%m%d%H%M%S").replace(tzinfo=UTC).timestamp()


def matching_profiles(path: Path, argo: list[tuple]) -> dict:
    """Count close collocations by platform, UTC time, and observation coordinates."""
    if not path.is_file():
        return {"status": "missing", "file": str(path)}
    with Dataset(path) as dataset:
        platforms = np.char.strip(chartostring(np.ma.filled(dataset["PLATFORM_NUMBER"][:], b"")).astype(str))
        time = np.ma.filled(dataset["JULD"][:].astype("f8"), np.nan) * 86400 + reference_timestamp(dataset)
        latitude = np.ma.filled(dataset["LATITUDE"][:].astype("f8"), np.nan)
        longitude = np.ma.filled(dataset["LONGITUDE"][:].astype("f8"), np.nan) % 360
    in_domain = np.isfinite(time) & (latitude >= -20) & (latitude <= 20) & (longitude >= 120) & (longitude <= 280)
    present = defaultdict(list)
    for platform, seconds, lat, lon in zip(
        platforms[in_domain], time[in_domain], latitude[in_domain], longitude[in_domain], strict=True,
    ):
        present[platform].append((seconds, lat, lon))
    matched = sum(
        any(
            abs(seconds - candidate_time) <= TIME_TOLERANCE_SECONDS
            and abs(lat - candidate_lat) <= POSITION_TOLERANCE_DEGREES
            and abs(lon - candidate_lon) <= POSITION_TOLERANCE_DEGREES
            for candidate_time, candidate_lat, candidate_lon in present.get(platform, ())
        )
        for platform, seconds, lat, lon in argo
    )
    date_range = [
        datetime.fromtimestamp(float(value), UTC).isoformat()
        for value in (time[in_domain].min(), time[in_domain].max())
    ]
    return {
        "status": "checked",
        "file": str(path),
        "matches": matched,
        "eligible_profiles": len(argo),
        "observation_time_range_utc_in_domain": date_range,
    }


def main() -> None:
    """Verify month-wise overlap in all v2 development and evaluation blocks."""
    records = []
    for row in baseline_manifest()["months"]:
        month = row["month"]
        split = split_for(month)
        if split not in CHECKED_SPLITS:
            continue
        with Dataset(row["argo_profiles"]) as dataset:
            keep = np.asarray(dataset["role_code"][:]) == 2
            argo = list(zip(*(
                np.asarray(dataset[key][:])[keep]
                for key in ("platform_number", "observation_time", "latitude", "longitude")
            ), strict=True))
        by_variable = {
            variable: matching_profiles(RAW / "cora" / f"OA_CORA5.2_{month}15_dat_{variable}.nc", argo)
            for variable in ("TEMP", "PSAL")
        }
        records.append({"month": month, "split": split, "matches": by_variable})
    summary = {}
    for split in CHECKED_SPLITS:
        summary[split] = {}
        for variable in ("TEMP", "PSAL"):
            matches = [row["matches"][variable] for row in records if row["split"] == split]
            checked = [row for row in matches if row["status"] == "checked"]
            summary[split][variable] = {
                "months_checked": len(checked),
                "months_missing": len(matches) - len(checked),
                "matched_profiles": sum(row["matches"] for row in checked),
                "profiles_checked": sum(row["eligible_profiles"] for row in checked),
            }
    report = {
        "scope": "Same nominal month source inventory; a match supplies evidence of observation reuse",
        "limitations": "Does not establish non-overlap for unmatched profiles or quantify their influence on CORA",
        "time_tolerance_seconds": TIME_TOLERANCE_SECONDS,
        "latitude_longitude_tolerance_degrees": POSITION_TOLERANCE_DEGREES,
        "summary": summary,
        "months": records,
    }
    write_json(DATASET / "audits/cora_e_overlap.json", report)
    for split, value in summary.items():
        print(split, value, flush=True)


if __name__ == "__main__":
    main()
