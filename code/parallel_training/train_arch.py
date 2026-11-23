#!/usr/bin/env python3
"""四模型统一训练入口：复用 train_parallel.train()，仅把模型工厂换成指定架构。

用法：
  PYTHONPATH=<ablate_site>:<python_pkgs>:<code>:<parallel_training> \
  ABLATE_MODE=no_smap /opt/conda/bin/python -u parallel_training/train_arch.py \
    --arch cnn --dataset-dir .../training_ready_v4 --run-dir .../v4A_cnn \
    --device cuda --workers 2 --validation-workers 1 --cpu-threads 1 \
    --amp --cache-gib 14 --epochs 200 --patience 201 --seed 20261120 --dim 48 --blocks 4

架构与参数映射（保持与 2026-11-20 四模型对照一致）：
  swin : SwinReconstructor(276, dim, blocks, heads, window)        --dim 48  --blocks 4 --heads 4 --window 8
  cnn  : ResidualCNN(width=dim, blocks=blocks)                     --dim 48  --blocks 4
  unet : UNetReconstructor(width=dim)                              --dim 24
  vit  : ViTReconstructor(embedding_dim=dim, blocks, heads, 0.0)   --dim 96  --blocks 4 --heads 4

注意：train_parallel.py 里模型构造是写死的，本脚本通过替换模块级符号实现，
不影响数据加载（worker 走 sitecustomize 的消融注入）。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

CODE_DIR = Path(__file__).resolve().parents[1]
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))
PARALLEL_DIR = Path(__file__).resolve().parent
if str(PARALLEL_DIR) not in sys.path:
    sys.path.insert(0, str(PARALLEL_DIR))

import train_parallel as tp  # noqa: E402
from residual_cnn.model import ResidualCNN  # noqa: E402
from train_swin_physics import SwinReconstructor  # noqa: E402
from unet_baseline.model import UNetReconstructor  # noqa: E402
from vit_baseline.model import ViTReconstructor  # noqa: E402

_ORIGINAL = SwinReconstructor


def main() -> None:
    wrapper = argparse.ArgumentParser(add_help=False)
    wrapper.add_argument("--arch", required=True, choices=["swin", "cnn", "unet", "vit"])
    known, rest = wrapper.parse_known_args()

    def factory(in_channels: int, dim: int, blocks: int, heads: int, window: int):
        if known.arch == "swin":
            return _ORIGINAL(in_channels, dim, blocks, heads, window)
        if known.arch == "cnn":
            return ResidualCNN(width=dim, blocks=blocks)
        if known.arch == "unet":
            return UNetReconstructor(width=dim)
        return ViTReconstructor(embedding_dim=dim, blocks=blocks, heads=heads, dropout=0.0)

    tp.SwinReconstructor = factory  # type: ignore[assignment]
    sys.argv = [sys.argv[0]] + rest
    print(f"[arch] {known.arch}", flush=True)
    tp.train(tp.parse_args())


if __name__ == "__main__":
    main()
