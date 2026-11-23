"""Regress native-grid CT/SA fields with a compact Vision Transformer."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class ViTReconstructor(nn.Module):
    """Apply global attention to 3 by 3 patches and decode all 60 depths.

    This model uses a learned position embedding and no classification token.
    Each independently initialized encoder layer attends to all 2,889 patches.
    The training entry seeds PyTorch before constructing the model.
    """

    def __init__(
        self,
        embedding_dim: int = 96,
        blocks: int = 4,
        heads: int = 4,
        dropout: float = 0.0,
    ) -> None:
        """Configure patch features and pre-normalized Transformer layers.

        Args:
            embedding_dim: Number of features per patch, divisible by heads.
            blocks: Number of independently initialized encoder layers.
            heads: Number of global self-attention heads per layer.
            dropout: Dropout probability, controlled by the recorded run seed.
        """
        super().__init__()
        if embedding_dim < 1 or heads < 1 or embedding_dim % heads or blocks < 1:
            raise ValueError("embedding-dim must be positive and divisible by heads; blocks must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        self.patch_embedding = nn.Conv2d(276, embedding_dim, kernel_size=3, stride=3)
        self.position_embedding = nn.Parameter(torch.empty(1, 27 * 107, embedding_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.blocks = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=embedding_dim,
                nhead=heads,
                dim_feedforward=embedding_dim * 4,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(blocks)
        )
        self.norm = nn.LayerNorm(embedding_dim)
        self.head = nn.ConvTranspose2d(embedding_dim, 120, kernel_size=3, stride=3)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return normalized CT/SA anomalies with shape [batch, 2, 60, 81, 321]."""
        if inputs.shape[1:] != (276, 81, 321):
            raise ValueError("ViT input must have shape [batch, 276, 81, 321]")
        embedded = cast(torch.Tensor, self.patch_embedding(inputs))
        tokens = embedded.flatten(2).transpose(1, 2) + self.position_embedding
        for block in self.blocks:
            tokens = cast(torch.Tensor, block(tokens))
        features = cast(torch.Tensor, self.norm(tokens)).transpose(1, 2).reshape(inputs.shape[0], -1, 27, 107)
        fields = cast(torch.Tensor, self.head(features))
        return fields.reshape(inputs.shape[0], 2, 60, 81, 321)
