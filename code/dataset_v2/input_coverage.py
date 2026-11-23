#!/usr/bin/env python3
"""逐月检查海表输入覆盖率（判断 2025 年是否出现输入数据缺口）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from server_training_data import TrainingDataset  # noqa: E402

DATASET = Path("/g3240101584lt/srtp-ocean/data/training_ready_v2")


def main() -> None:
    out_path = Path(sys.argv[1])
    source = TrainingDataset(DATASET)
    months = [m for m in source.records if "202201" <= m <= "202506"]
    result = {}
    for month in sorted(months):
        sample = source.load(month)
        masks = np.asarray(sample["surface_mask"])          # (n_sources*3, 81, 321)
        record = source.records[month]
        names = list(record["surface_history_t_minus_2_to_t"].keys())
        entry = {}
        for i, name in enumerate(names):
            for j, label in enumerate(("t-2", "t-1", "t")):
                idx = i * 3 + j
                entry[f"{name}:{label}"] = float(masks[idx].mean())
        result[month] = entry
        print(month, "ok", flush=True)
    out_path.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print("WROTE", out_path, flush=True)


if __name__ == "__main__":
    main()
