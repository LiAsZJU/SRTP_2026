#!/usr/bin/env python3
"""Build 数据集 v4（方案 A：训练 2005-01–2019-11，验证 2020–2021，测试 2022–2024，前向 2025H1）。

与 build_dataset.py（v2）的区别：
  - split 表按方案 A 重划；
  - 统计量（背景/归一化）只用新训练期（179 个月）拟合；
  - v4 全流程不使用 SMAP（训练时 ABLATE_MODE=no_smap），因此对 SMAP 只记录窗口界、不强制截断；
  - SMAP 边界文件复用 v2 的 smap_boundary；
  - 不覆盖 v2，输出到 training_ready_v4。

用法：
  python dataset_v2/build_dataset_v4.py fit      # 重拟合背景 + 归一化
  python dataset_v2/build_dataset_v4.py publish  # 生成 manifest + 审计
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
from netCDF4 import Dataset, num2date

from prepare_server_training_data import RAW, build_background, build_normalization, month_add

DATASET = RAW / "training_ready_v4"
BASELINE = RAW / "training_ready_v1"
V2 = RAW / "training_ready_v2"
VERSION = "training_ready_v4_A_2005_2019"
TRAIN_START = "200501"
TRAIN_END = "201911"
CUTOFFS = {
    "training": "2020-01-01T00:00:00Z",
    "validation": "2022-01-01T00:00:00Z",
    "test_locked": "2025-01-01T00:00:00Z",
}
EXPECTED_COUNTS = {
    "training": 179,
    "isolation": 1,
    "validation": 24,
    "test_locked": 36,
    "forward_test": 6,
}
BOUNDARY_SMAP = {"202312", "202412"}
BOUNDARY_DIR = V2 / "smap_boundary"
# v4 不使用 SMAP（训练/验证/测试输入统一置零），只记录窗口界，不做截断强制。
ENFORCE_SMAP_CUTOFF = False


def split_for(month: str) -> str:
    """方案 A 的时间块划分。"""
    if month < TRAIN_START:
        return "archive"
    if month <= TRAIN_END:
        return "training"
    if month == "201912":
        return "isolation"
    if month <= "202112":
        return "validation"
    if month <= "202412":
        return "test_locked"
    return "forward_test"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    temporary.replace(path)


def baseline_manifest() -> dict:
    manifest = json.loads((BASELINE / "training_manifest.json").read_text(encoding="utf-8"))
    months = [row["month"] for row in manifest["months"]]
    if months != [month_add("200501", offset) for offset in range(246)]:
        raise ValueError("Unexpected source manifest months")
    return manifest


def fit_statistics() -> None:
    months = [row["month"] for row in baseline_manifest()["months"]]
    training = [month for month in months if split_for(month) == "training"]
    if len(training) != EXPECTED_COUNTS["training"]:
        raise ValueError(f"Unexpected training period: {len(training)}")
    stats = DATASET / "statistics"
    stats.mkdir(parents=True, exist_ok=True)
    targets = (stats / "training_background.nc", stats / "normalization.json")
    if any(path.exists() for path in targets):
        raise FileExistsError("Statistics targets already exist")
    background = build_background(months, fitted_months=training, destination=DATASET, version=VERSION)
    build_normalization(months, background, fitted_months=training, destination=DATASET, version=VERSION)


def evaluate_cohorts(rows: list[dict]) -> dict:
    validation_platforms = set()
    cases = ("validation", "test_locked", "forward_test")
    samples: dict[str, list[tuple[str, np.ndarray, np.ndarray]]] = {split: [] for split in cases}
    for row in rows:
        split = row["split"]
        if split not in samples:
            continue
        with Dataset(row["argo_profiles"]) as dataset:
            is_evaluation = np.asarray(dataset["role_code"][:]) == 2
            platforms = np.asarray(dataset["platform_number"][:], dtype=str)[is_evaluation]
            valid_levels = np.asarray(dataset["valid"][:])[is_evaluation].sum(axis=1)
        if split == "validation":
            validation_platforms.update(platforms.tolist())
        samples[split].append((row["month"], platforms, valid_levels))
    cohorts: dict[str, dict] = {}
    for split, records in samples.items():
        platforms = np.concatenate([values for _, values, _ in records])
        counts = np.concatenate([counts for _, _, counts in records])
        all_platforms = set(platforms.tolist())
        unseen = all_platforms - validation_platforms
        is_unseen = np.isin(platforms, sorted(unseen))
        cohorts[split] = {
            "months": [month for month, _, _ in records],
            "profiles": int(platforms.size),
            "valid_level_points_per_variable_before_normalization": int(counts.sum()),
            "platform_count": len(all_platforms),
            "validation_seen_platform_count": len(all_platforms & validation_platforms),
            "validation_unseen_platform_count": len(unseen),
            "validation_unseen_profiles": int(is_unseen.sum()),
            "validation_unseen_platforms": sorted(unseen),
        }
    return {"validation_platforms": sorted(validation_platforms), "cohorts": cohorts}


def publish_manifest() -> None:
    manifest = baseline_manifest()
    rows = manifest["months"]
    normalization = DATASET / "statistics/normalization.json"
    background = DATASET / "statistics/training_background.nc"
    split_counts = Counter()
    files = {normalization, background}
    smap_ends = {}
    other_product_ends = {}
    monthly_audit = []
    for row in rows:
        month = row["month"]
        split = row["split"] = split_for(month)
        row["background"] = str(background)
        split_counts[split] += 1
        files.update(Path(row[key]) for key in ("cora", "argo_grid", "argo_profiles"))
        for source, items in row["surface_history_t_minus_2_to_t"].items():
            if [item["month"] for item in items] != [month_add(month, i) for i in (-2, -1, 0)]:
                raise ValueError(f"{month}: incomplete {source} history")
            for item in items:
                key = item["month"]
                if source == "smap":
                    if key in BOUNDARY_SMAP:
                        item["path"] = str(BOUNDARY_DIR / f"smap_{key}.nc")
                    path = Path(item["path"])
                    if path not in smap_ends:
                        with Dataset(path) as dataset:
                            smap_ends[path] = getattr(dataset, "window_support_end", None)
                    end = smap_ends[path]
                    if (ENFORCE_SMAP_CUTOFF and split in CUTOFFS and end is not None
                            and datetime.fromisoformat(end) > datetime.fromisoformat(CUTOFFS[split])):
                        raise ValueError(f"{month}: {path.name} exceeds {split} cutoff")
                elif split in CUTOFFS:
                    path = Path(item["path"])
                    if path not in other_product_ends:
                        with Dataset(path) as dataset:
                            time = dataset["time"]
                            bound = dataset[time.bounds][0, 1]
                            end = num2date(bound, time.units, getattr(time, "calendar", "standard"))
                            other_product_ends[path] = end.isoformat() + "+00:00"
                    if datetime.fromisoformat(other_product_ends[path]) > datetime.fromisoformat(CUTOFFS[split]):
                        raise ValueError(f"{month}: {path.name} exceeds the nominal {split} cutoff")
                files.add(Path(item["path"]))
        if split in CUTOFFS:
            monthly_audit.append({
                "month": month,
                "split": split,
                "cutoff_utc": CUTOFFS[split],
                "nominal_product_end_utc": {
                    key: [other_product_ends[Path(entry["path"])]
                          for entry in row["surface_history_t_minus_2_to_t"][key]]
                    for key in ("oisst", "adt", "era5_u10", "era5_v10")
                },
                "smap_inputs": [
                    {"month": entry["month"],
                     "window_support_end": smap_ends[Path(entry["path"])],
                     "boundary_rebuilt": entry["month"] in BOUNDARY_SMAP}
                    for entry in row["surface_history_t_minus_2_to_t"]["smap"]
                ],
            })
    if dict(split_counts) != EXPECTED_COUNTS:
        raise ValueError(f"Incorrect split counts: {split_counts}")
    missing = sorted(str(path) for path in files if not path.is_file())
    if missing:
        raise FileNotFoundError(missing)
    training = [row["month"] for row in rows if row["split"] == "training"]
    stats = json.loads(normalization.read_text(encoding="utf-8"))
    with Dataset(background) as dataset:
        if stats["fitted_months"] != training or dataset.fitted_months.split(",") != training:
            raise ValueError("Statistics do not match the training months")
    for key, stat in stats["statistics"].items():
        if not all(np.isfinite(stat[name]).all() for name in ("mean", "std")):
            raise ValueError(f"Non-finite statistics for {key}")
        if not np.all(np.asarray(stat["count"]) > 0):
            raise ValueError(f"No usable training support for {key}")
    cohorts = evaluate_cohorts(rows)
    write_json(DATASET / "evaluation_platforms.json", cohorts)
    write_json(DATASET / "audits/time_boundaries.json",
               {"cutoffs_utc": CUTOFFS, "smap_cutoff_enforced": ENFORCE_SMAP_CUTOFF,
                "months": monthly_audit})
    manifest.update({
        "schema": "swin_training_manifest_v2",
        "processing_version": VERSION,
        "normalization": str(normalization),
        "source_manifest": str(BASELINE / "training_manifest.json"),
        "split_counts": EXPECTED_COUNTS,
        "evaluation_platforms": str(DATASET / "evaluation_platforms.json"),
        "smap_time_boundary_audit": str(DATASET / "audits/time_boundaries.json"),
    })
    write_json(DATASET / "training_manifest.json", manifest)
    print("DATASET_READY", EXPECTED_COUNTS, "unique_paths", len(files), flush=True)
    for split, cohort in cohorts["cohorts"].items():
        print(split, {k: v for k, v in cohort.items() if k != "months" and not isinstance(v, list)}, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["fit", "publish"])
    args = parser.parse_args()
    if args.stage == "fit":
        fit_statistics()
    else:
        publish_manifest()


if __name__ == "__main__":
    main()
