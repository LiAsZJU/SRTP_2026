"""Unit tests for v4 Argo C Context Dropout and Swin/CNN compatibility."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch

PARALLEL_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = PARALLEL_DIR.parent
for path_item in (CODE_DIR, PARALLEL_DIR):
    if str(path_item) not in sys.path:
        sys.path.insert(0, str(path_item))

from context_dropout import (  # noqa: E402
    ALL_CONTEXT_SLICE,
    SMAP_MASK_SLICE,
    SMAP_VALUE_SLICE,
    SURFACE_CHANNEL_SLICE,
    ContextDropoutConfig,
    apply_context_dropout,
    assert_no_smap_input,
    zero_all_context,
)
from residual_cnn.model import ResidualCNN  # noqa: E402
from train_swin_physics import SwinReconstructor  # noqa: E402


def make_synthetic_v4_input(batch: int = 1, height: int = 81, width: int = 321) -> torch.Tensor:
    """Create a synthetic [B, 276, H, W] input tensor with SMAP strictly zeroed."""
    x = torch.randn((batch, 276, height, width), dtype=torch.float32)
    # Surface masks (channels 15:30) and SMAP zeroing (12:15 and 27:30).
    x[:, 15:30] = 1.0
    x[:, SMAP_VALUE_SLICE] = 0.0
    x[:, SMAP_MASK_SLICE] = 0.0
    # Valid Argo C mask on a subset of spatial cells, aligned with offset_mask.
    valid_spatial = torch.zeros((batch, 1, height, width), dtype=torch.float32)
    valid_spatial[:, :, 10:70, 20:300] = 1.0
    x[:, 150:270] = valid_spatial
    x[:, 273:276] = valid_spatial
    x[:, 30:150] *= valid_spatial
    x[:, 270:273] *= valid_spatial
    return x


class ContextDropoutTests(unittest.TestCase):
    """Verify channel safety, non-mutation, determinism, and Swin/CNN forward+backward."""

    def test_no_smap_guard(self) -> None:
        x = make_synthetic_v4_input()
        assert_no_smap_input(x)
        x_bad = x.clone()
        x_bad[:, 12, 0, 0] = 0.5
        with self.assertRaises(RuntimeError):
            assert_no_smap_input(x_bad)

    def test_surface_untouched_and_cache_not_mutated(self) -> None:
        x = make_synthetic_v4_input()
        snapshot = x.clone()
        gen = torch.Generator(device="cpu").manual_seed(42)
        for mode in ("pointwise", "block", "full_drop", "mixed"):
            cfg = ContextDropoutConfig(mode=mode, p_clean=0.0)
            aug, kept_frac, regime = apply_context_dropout(x, cfg, gen)
            self.assertTrue(torch.equal(x, snapshot), f"Input mutated in mode={mode}")
            self.assertTrue(
                torch.equal(aug[:, SURFACE_CHANNEL_SLICE], x[:, SURFACE_CHANNEL_SLICE]),
                f"Surface channels changed in mode={mode}",
            )
            self.assertGreaterEqual(kept_frac, 0.0)
            self.assertLessEqual(kept_frac, 1.0)
            self.assertIn(regime, ("pointwise", "block", "full_drop"))

    def test_dropped_cells_zero_offsets_and_masks_together(self) -> None:
        x = make_synthetic_v4_input()
        cfg = ContextDropoutConfig(mode="pointwise", min_keep=0.3, max_keep=0.3)
        gen = torch.Generator(device="cpu").manual_seed(123)
        aug, kept_frac, regime = apply_context_dropout(x, cfg, gen)
        self.assertEqual(regime, "pointwise")
        self.assertGreater(kept_frac, 0.15)
        self.assertLess(kept_frac, 0.45)

        # Wherever CT context_mask was zeroed, offsets and offset_masks must also be zero.
        kept_mask = aug[:, 150:270].amax(dim=1) > 0
        offset_nonzero = aug[:, 270:276].abs().amax(dim=1) > 0
        self.assertTrue(bool((offset_nonzero <= kept_mask).all()))

    def test_full_drop_and_zero_all_context(self) -> None:
        x = make_synthetic_v4_input()
        cfg = ContextDropoutConfig(mode="full_drop")
        aug, kept_frac, regime = apply_context_dropout(x, cfg)
        self.assertEqual(regime, "full_drop")
        self.assertEqual(kept_frac, 0.0)
        self.assertEqual(float(aug[:, ALL_CONTEXT_SLICE].abs().max().item()), 0.0)
        z = zero_all_context(x)
        self.assertTrue(torch.equal(aug, z))

    def test_determinism(self) -> None:
        x = make_synthetic_v4_input()
        cfg = ContextDropoutConfig(mode="mixed")
        g1 = torch.Generator(device="cpu").manual_seed(999)
        g2 = torch.Generator(device="cpu").manual_seed(999)
        a1, f1, r1 = apply_context_dropout(x, cfg, g1)
        a2, f2, r2 = apply_context_dropout(x, cfg, g2)
        self.assertEqual(f1, f2)
        self.assertEqual(r1, r2)
        self.assertTrue(torch.equal(a1, a2))

    def test_swin_and_cnn_forward_backward_smoke(self) -> None:
        x = make_synthetic_v4_input()
        cfg = ContextDropoutConfig(mode="block")
        gen = torch.Generator(device="cpu").manual_seed(7)
        aug, _, _ = apply_context_dropout(x, cfg, gen)
        for model in (
            SwinReconstructor(276, dim=48, blocks=4, heads=4, window=8),
            ResidualCNN(width=48, blocks=4),
        ):
            out = model(aug)
            self.assertEqual(tuple(out.shape), (1, 2, 60, 81, 321))
            loss = out.square().mean()
            loss.backward()


if __name__ == "__main__":
    unittest.main()
