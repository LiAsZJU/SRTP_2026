#!/usr/bin/env python3
"""杈撳叆婧愭秷铻嶅寘瑁呭櫒锛氫笉淇敼 train_parallel.py锛岄€氳繃鏇挎崲 model_input 瀹炵幇鈥滈€氶亾缃浂鈥濄€?
涓轰粈涔堢疆闆惰€屼笉鏄垹閫氶亾锛?  train_parallel.py 閲?SwinReconstructor 鐨勮緭鍏ラ€氶亾鏁版槸鍐欐鐨?276锛?  缃浂鍙互淇濇寔妯″瀷缁撴瀯涓庡弬鏁伴噺瀹屽叏涓€鑷达紙288,024锛夛紝鏄渶骞插噣鐨勬帶鍒跺彉閲忋€?
鐢ㄦ硶锛堝湪 code 鐩綍涓嬶級锛?  python parallel_training/ablate_input.py --ablate no_smap --run-dir ... <鍏朵綑鍙傛暟鍘熸牱閫忎紶>

娑堣瀺妯″紡锛?  full          鍏ㄩ儴閫氶亾锛堝熀绾匡級
  no_smap       缃浂 SMAP 鐨勫€?鎺╄啘
  no_wind       缃浂 ERA5 u10/v10
  no_adt        缃浂 ADT
  oisst_only    鍙繚鐣?OISST锛堝叾浣欐捣琛ㄦ簮鍏ㄩ儴缃浂锛?  no_context    缃浂 C 缁勫唴閮ㄤ笂涓嬫枃锛堝€?鎺╄啘锛?40 閫氶亾锛?  context_keep  鎸夋瘮渚嬮殢鏈轰繚鐣?C 缁勯敋鐐癸紙閰嶅悎 --context-keep 0.5/0.1/0锛?
閫氶亾甯冨眬锛堜笌 model_input 涓€鑷达級锛?  娴疯〃婧愬€?0-14 | 娴疯〃鎺╄啘 15-29 | C 缁?CT/SA 30-149 | C 缁勬帺鑶?150-269 | 鍋忕Щ 270-272 | 鍋忕Щ鎺╄啘 273-275
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import train_parallel as tp  # noqa: E402

SOURCE_ORDER = ("oisst", "adt", "era5_u10", "era5_v10", "smap")
CONTEXT_VALUES = slice(30, 150)
CONTEXT_MASKS = slice(150, 270)


def source_channels(name: str) -> list[int]:
    i = SOURCE_ORDER.index(name)
    return [i * 3 + j for j in range(3)] + [15 + i * 3 + j for j in range(3)]


def zero_channels(ablate: str) -> list[int]:
    all_surface = list(range(30))
    if ablate == "full":
        return []
    if ablate == "no_smap":
        return source_channels("smap")
    if ablate == "no_wind":
        return source_channels("era5_u10") + source_channels("era5_v10")
    if ablate == "no_adt":
        return source_channels("adt")
    if ablate == "oisst_only":
        keep = set(source_channels("oisst"))
        return [i for i in all_surface if i not in keep]
    if ablate in ("adt_only", "wind_only", "smap_only"):
        names = {"adt_only": ("adt",), "wind_only": ("era5_u10", "era5_v10"), "smap_only": ("smap",)}[ablate]
        keep = set()
        for name in names:
            keep.update(source_channels(name))
        return [i for i in all_surface if i not in keep]
    if ablate == "no_context":
        return list(range(30, 270))
    if ablate == "no_surface":
        return list(range(0, 30))
    if ablate == "offsets_only":
        return list(range(0, 270))
    if ablate == "context_keep":
        return []
    raise ValueError(ablate)


def build_channels(sample: dict) -> np.ndarray:
    return np.concatenate(
        (
            sample["surface"],
            sample["surface_mask"],
            sample["context"].reshape(-1, 81, 321),
            sample["context_mask"].reshape(-1, 81, 321),
            sample["context_offsets"],
            sample["context_offset_mask"],
        ),
        axis=0,
    )


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--ablate", default="full",
                         choices=["full", "no_smap", "no_wind", "no_adt", "oisst_only", "no_context", "no_surface", "offsets_only", "adt_only", "wind_only", "smap_only", "context_keep"])
    wrapper.add_argument("--context-keep", type=float, default=1.0)
    wrapper.add_argument("--context-seed", type=int, default=20261120)
    known, rest = wrapper.parse_known_args()

    zeros = zero_channels(known.ablate)
    mode = known.ablate
    keep_frac = known.context_keep
    seed = known.context_seed

    def patched(sample: dict, device: torch.device) -> torch.Tensor:
        channels = build_channels(sample).copy()
        if zeros:
            channels[zeros] = 0.0
        if mode == "context_keep" and keep_frac < 1.0:
            mask_ct = channels[150].reshape(60, 81, 321)
            cells = np.argwhere(mask_ct.any(axis=0))
            if cells.size:
                rng = np.random.default_rng(seed + int(sample["month"]))
                k = max(1, int(round(cells.shape[0] * keep_frac)))
                chosen = cells[rng.permutation(cells.shape[0])[:k]]
                iy, ix = chosen[:, 0], chosen[:, 1]
                values = channels[CONTEXT_VALUES].reshape(2, 60, 81, 321).copy()
                values[:, :, iy, ix] = 0
                channels[CONTEXT_VALUES] = values.reshape(120, 81, 321)
                masks = channels[CONTEXT_MASKS].reshape(2, 60, 81, 321).copy()
                masks[:, :, iy, ix] = 0
                channels[CONTEXT_MASKS] = masks.reshape(120, 81, 321)
        return torch.from_numpy(channels[None]).to(device=device, dtype=torch.float32)

    tp.model_input = patched
    sys.argv = [sys.argv[0]] + rest
    print(f"[ablate] mode={mode} keep={keep_frac} zeroed_channels={len(zeros)}", flush=True)
    tp.train(tp.parse_args())


if __name__ == "__main__":
    main()
