"""在每个 Python 进程（含 DataLoader spawn worker）启动时注入输入消融。

为什么需要它：
  之前用“在主进程 monkey-patch train_parallel.model_input”的方式做消融，
  但 DataLoader 使用 spawn 时会重新导入模块，主进程的 patch **不会**进入 worker，
  因此消融实际没有生效（表现为 offsets_only ≈ 基线）。
  本文件由 site 模块在每个解释器启动时自动导入，workers 也会执行，从而保证消融对训练/验证都生效。

用法：
  ABLATE_MODE=no_smap PYTHONPATH=<本目录>:$PYTHONPATH python parallel_training/train_parallel.py ...
"""

from __future__ import annotations

import os

_MODE = os.environ.get("ABLATE_MODE", "full").strip()
_SURF = {
    "oisst": list(range(0, 3)),
    "adt": list(range(3, 6)),
    "era5_u10": list(range(6, 9)),
    "era5_v10": list(range(9, 12)),
    "smap": list(range(12, 15)),
}


def _zero_surface(sample: dict, sources: list[str]) -> None:
    idx = [i for name in sources for i in _SURF[name]]
    sample["surface"][idx] = 0
    sample["surface_mask"][idx] = 0


def _apply(sample: dict, mode: str) -> None:
    if mode == "full":
        return
    if mode == "no_smap":
        _zero_surface(sample, ["smap"])
    elif mode == "no_wind":
        _zero_surface(sample, ["era5_u10", "era5_v10"])
    elif mode == "no_adt":
        _zero_surface(sample, ["adt"])
    elif mode == "oisst_only":
        _zero_surface(sample, ["adt", "era5_u10", "era5_v10", "smap"])
    elif mode == "adt_only":
        _zero_surface(sample, ["oisst", "era5_u10", "era5_v10", "smap"])
    elif mode == "wind_only":
        _zero_surface(sample, ["oisst", "adt", "smap"])
    elif mode == "smap_only":
        _zero_surface(sample, ["oisst", "adt", "era5_u10", "era5_v10"])
    elif mode == "no_surface":
        _zero_surface(sample, list(_SURF))
    elif mode == "no_context":
        sample["context"][:] = 0
        sample["context_mask"][:] = 0
    elif mode == "offsets_only":
        _zero_surface(sample, list(_SURF))
        sample["context"][:] = 0
        sample["context_mask"][:] = 0
    else:
        raise ValueError(f"未知 ABLATE_MODE: {mode}")


if _MODE != "full":
    try:
        import server_training_data as _std

        if not getattr(_std.TrainingDataset.load, "__ablate_patched__", False):
            _original_load = _std.TrainingDataset.load

            def _patched_load(self, month):
                sample = _original_load(self, month)
                _apply(sample, os.environ.get("ABLATE_MODE", "full"))
                return sample

            _patched_load.__ablate_patched__ = True
            _std.TrainingDataset.load = _patched_load
    except Exception as exc:  # pragma: no cover - 启动期失败不应阻断训练
        print(f"[ablate-site] 注入失败: {exc}", flush=True)
