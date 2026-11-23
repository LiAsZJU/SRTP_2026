#!/usr/bin/env python3
"""Sparse Argo-anchor conditioning for monthly 3-D CT/SA reconstruction.

这是「稀疏锚点」思路的原型实现，**不改动任何现有文件**：
  - 网格输入仍来自海表历史场（OISST/ADT/ERA5/SMAP × 3 个月）与 C 组位置偏移；
  - C 组 Argo 不再作为 240 个稠密通道塞进卷积，而是抽成 **变长点集 token**
    （每个锚点 = 一根剖面：60 层的 CT/SA 异常 + 有效掩膜 + 经纬度/时间位置）；
  - 网格 token 通过 **cross-attention** 查询锚点 token，相当于让网络自己学一个
    「以观测为条件的客观分析算子」；
  - 训练时按 `anchor_keep` 随机丢弃整根剖面，用来模拟不同观测密度，
    从而可以画出「锚点数 → 重建误差」的信息曲线。

依赖：torch、torchvision、numpy；复用 train_swin_physics.SwinBlock。
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from train_swin_physics import SwinBlock

DEPTH = 60
GRID_H = 81
GRID_W = 321
PATCH = 3
TOK_H = GRID_H // PATCH   # 27
TOK_W = GRID_W // PATCH   # 107


def grid_input(sample: dict) -> torch.Tensor:
    """拼出网格侧输入（不含 C 组稠密上下文）。

    通道 = 海表场(15) + 海表掩膜(15) + C 组位置偏移(3) + 偏移掩膜(3) = 36。
    """
    channels = np.concatenate(
        (
            sample["surface"],
            sample["surface_mask"],
            sample["context_offsets"],
            sample["context_offset_mask"],
        ),
        axis=0,
    )
    return torch.from_numpy(channels[None]).float()


def anchors_from_sample(sample: dict) -> dict[str, torch.Tensor]:
    """把 C 组 Argo 的稀疏网格占用抽成变长锚点 token。

    现有数据里 C 组只以「稀疏网格占用」的形式存在（valid 比例约 2.6%），
    因此这里把每个有效网格列近似为一个剖面锚点：取该列 60 层的 CT/SA 异常。

    Returns:
        values: (A, 60, 2) 归一化异常，无效层置 0
        mask:   (A, 60)    层有效掩膜
        feats:  (A, 5)     归一化网格位置 + C 组纬度/经度/时间偏移
        pad:    (A,)       False
    """
    valid = np.asarray(sample["context_mask"])[0] > 0          # (60, 81, 321)
    cells = valid.any(axis=0)                                   # (81, 321)
    iy, ix = np.nonzero(cells)
    if iy.size == 0:
        a = torch.zeros(1, DEPTH, 2)
        m = torch.zeros(1, DEPTH)
        f = torch.zeros(1, 5)
        return {"values": a, "mask": m, "feats": f, "pad": torch.ones(1, dtype=torch.bool)}

    ct = np.asarray(sample["context"])[0][:, iy, ix].T          # (A, 60)
    sa = np.asarray(sample["context"])[1][:, iy, ix].T
    m = valid[:, iy, ix].T.astype("f4")                          # (A, 60)
    off = np.asarray(sample["context_offsets"])                  # (3, 81, 321)
    feats = np.stack(
        (
            iy / (GRID_H - 1),
            ix / (GRID_W - 1),
            off[0][iy, ix],
            off[1][iy, ix],
            off[2][iy, ix],
        ),
        axis=1,
    ).astype("f4")

    values = np.stack((ct * m, sa * m), axis=-1).astype("f4")
    values[~np.isfinite(values)] = 0.0
    feats[~np.isfinite(feats)] = 0.0
    return {
        "values": torch.from_numpy(values),
        "mask": torch.from_numpy(m),
        "feats": torch.from_numpy(feats),
        "pad": torch.zeros(iy.size, dtype=torch.bool),
    }


def drop_anchors(anchors: dict[str, torch.Tensor], keep: float, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """按比例随机丢弃整根锚点剖面（模拟不同观测密度，至少保留 1 根）。"""
    if keep >= 1.0 or anchors["pad"].numel() <= 1:
        return anchors
    n = anchors["pad"].numel()
    keep_n = max(1, int(round(n * keep)))
    order = torch.randperm(n, generator=generator)
    kept = order[:keep_n].sort().values
    return {
        "values": anchors["values"][kept],
        "mask": anchors["mask"][kept],
        "feats": anchors["feats"][kept],
        "pad": anchors["pad"][kept],
    }


class SparseAnchorModel(nn.Module):
    """Swin 网格主干 + 稀疏锚点 cross-attention 条件化。

    形状约定：
        grid:   (B, C_in, 81, 321)
        values: (B, A, 60, 2)   mask: (B, A, 60)   feats: (B, A, 5)   pad: (B, A) bool
        output: (B, 2, 60, 81, 321)
    """

    def __init__(
        self,
        in_channels: int,
        dim: int = 48,
        blocks: int = 4,
        heads: int = 4,
        window: int = 8,
        anchor_hidden: int = 128,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim 必须能被 heads 整除")
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=PATCH, stride=PATCH)
        # 锚点 token 输入 = CT(60) + SA(60) + mask(60) + pos(5) = 185
        self.anchor_mlp = nn.Sequential(
            nn.Linear(2 * DEPTH + DEPTH + 5, anchor_hidden),
            nn.GELU(),
            nn.LayerNorm(anchor_hidden),
            nn.Linear(anchor_hidden, dim),
        )
        self.cross_q_norm = nn.LayerNorm(dim)
        self.cross_kv_norm = nn.LayerNorm(dim)
        self.cross = nn.MultiheadAttention(dim, heads, batch_first=True, dropout=dropout)
        self.grid_blocks = nn.ModuleList(
            SwinBlock(dim, heads, window, shifted=bool(index % 2)) for index in range(blocks)
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.ConvTranspose2d(dim, 2 * DEPTH, kernel_size=PATCH, stride=PATCH)
        self.anchor_gate = nn.Parameter(torch.zeros(1))

    def forward(
        self,
        grid: torch.Tensor,
        values: torch.Tensor,
        mask: torch.Tensor,
        feats: torch.Tensor,
        pad: torch.Tensor,
    ) -> torch.Tensor:
        batch = grid.shape[0]
        feat = self.patch_embed(grid)                                   # (B, dim, 27, 107)
        tokens = feat.flatten(2).transpose(1, 2)                        # (B, N, dim)

        flat = torch.cat((values.reshape(batch, values.shape[1], -1),
                          mask,
                          feats), dim=-1)                               # (B, A, 185)
        anchor_tokens = self.anchor_mlp(flat)                           # (B, A, dim)

        if (~pad).sum() > 0:
            query = self.cross_q_norm(tokens)
            key = self.cross_kv_norm(anchor_tokens)
            attended, _ = self.cross(query, key, key, key_padding_mask=pad, need_weights=False)
            tokens = tokens + torch.sigmoid(self.anchor_gate) * attended

        h = tokens.transpose(1, 2).reshape(batch, tokens.shape[-1], TOK_H, TOK_W)
        for block in self.grid_blocks:
            h = block(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        h = self.norm(h.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        out = self.head(h)
        return out.reshape(batch, 2, DEPTH, GRID_H, GRID_W)


def build_batch(sample: dict, keep: float = 1.0, generator: torch.Generator | None = None) -> dict[str, torch.Tensor]:
    """把 TrainingDataset.load() 的单月样本整理成模型输入与锚点输入。"""
    anchors = drop_anchors(anchors_from_sample(sample), keep, generator)
    return {
        "grid": grid_input(sample),
        "values": anchors["values"][None],
        "mask": anchors["mask"][None],
        "feats": anchors["feats"][None],
        "pad": anchors["pad"][None],
    }
