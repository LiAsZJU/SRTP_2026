#!/usr/bin/env python3
"""Print compact schemas for the uploaded training datasets."""
from pathlib import Path

import json
import numpy as np
import xarray as xr


ROOT = Path("/g3240101584lt/srtp-ocean/data")
FILES = [
    ROOT / "cora/OA_CORA5.2_20180115_fld_TEMP.nc",
    ROOT / "cora/OA_CORA5.2_20180115_fld_PSAL.nc",
    ROOT / "oisst/months/201801/oisst_201801.nc",
    ROOT / "adt/months/201801/adt_201801.nc",
    ROOT / "era5/months/201801/era5_201801.nc",
    ROOT / "smap/months/201801/smap_201801.nc",
    ROOT / "argo/outputs/full_200501_202506_v1/months/201801/profiles_00001.nc",
]


for path in FILES:
    with xr.open_dataset(path, engine="h5netcdf", decode_times=False) as dataset:
        print(f"\nFILE {path}")
        print("DIMS", dict(dataset.sizes))
        for name, variable in dataset.variables.items():
            print("VAR", name, str(variable.dtype), variable.dims,
                  variable.attrs.get("units"), variable.attrs.get("standard_name"))

with xr.open_dataset(FILES[0], engine="h5netcdf", decode_times=False) as dataset:
    depth = dataset.depth.values
    selected = depth[(depth >= 5) & (depth <= 500)]
    print("\nCORA_TARGET_DEPTH", selected.size, selected.tolist())

with xr.open_dataset(FILES[-1], engine="h5netcdf", decode_times=False) as dataset:
    print("ARGO_DEPTH", dataset.depth.values.tolist())
    for name in ("role", "data_mode", "valid", "in_template_footprint"):
        values, counts = np.unique(dataset[name].values, return_counts=True)
        print("ARGO_VALUES", name, list(zip(values.tolist(), counts.tolist())))

for source, variable, mask in [
    ("oisst", "sst", "sst_valid"),
    ("adt", "adt", "adt_valid"),
    ("era5", "u10", "u10_valid"),
    ("era5", "v10", "v10_valid"),
    ("smap", "sss_smap", "sss_smap_valid"),
]:
    path = ROOT / source / "months/201801" / f"{source}_201801.nc"
    with xr.open_dataset(path, engine="h5netcdf", decode_times=False) as dataset:
        values = dataset[variable].values
        valid = dataset[mask].values.astype(bool)
        print("MASK", source, variable, int(valid.sum()), valid.size,
              int(np.isfinite(values).sum()), int((valid & np.isfinite(values)).sum()))

collection = json.loads((ROOT / "argo/outputs/argo_complete_v2/collection.json").read_text())
print("COLLECTION_MONTHS", len(collection["months"]), min(collection["months"]), max(collection["months"]))
