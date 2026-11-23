"""Dynamic Argo C Context Dropout for training-time 3D ocean reconstruction."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch

SURFACE_CHANNEL_SLICE = slice(0, 30)
SMAP_VALUE_SLICE = slice(12, 15)
SMAP_MASK_SLICE = slice(27, 30)
CONTEXT_VALUE_SLICE = slice(30, 150)
CONTEXT_MASK_SLICE = slice(150, 270)
CONTEXT_OFFSET_SLICE = slice(270, 276)
ALL_CONTEXT_SLICE = slice(30, 276)

VALID_CONTEXT_DROPOUT_MODES = ("none", "mixed", "pointwise", "block", "full_drop")


@dataclass(frozen=True)
class ContextDropoutConfig:
    """Configuration for training-time Argo C context dropout.

    Attributes:
        mode: Dropout regime selector: ``none``, ``mixed``, ``pointwise``, ``block``, or ``full_drop``.
        p_clean: Probability of keeping all Argo C observations unchanged in ``mixed`` mode.
        p_full: Probability (conditional on applying dropout in ``mixed`` mode) of full Argo C blackout.
        p_block: Probability (conditional on applying dropout in ``mixed`` mode) of spatial block cutout.
        min_keep: Minimum fraction of spatial cells retained in ``pointwise`` mode.
        max_keep: Maximum fraction of spatial cells retained in ``pointwise`` mode.
        num_blocks_min: Minimum number of rectangular cutout patches in ``block`` mode.
        num_blocks_max: Maximum number of rectangular cutout patches in ``block`` mode.
        block_h_min: Minimum latitude patch height (grid cells) in ``block`` mode.
        block_h_max: Maximum latitude patch height (grid cells) in ``block`` mode.
        block_w_min: Minimum longitude patch width (grid cells) in ``block`` mode.
        block_w_max: Maximum longitude patch width (grid cells) in ``block`` mode.
    """

    mode: str = "mixed"
    p_clean: float = 0.15
    p_full: float = 0.20
    p_block: float = 0.35
    min_keep: float = 0.10
    max_keep: float = 0.85
    num_blocks_min: int = 1
    num_blocks_max: int = 3
    block_h_min: int = 10
    block_h_max: int = 28
    block_w_min: int = 16
    block_w_max: int = 48

    def __post_init__(self) -> None:
        """Validate probability and spatial geometry bounds."""
        if self.mode not in VALID_CONTEXT_DROPOUT_MODES:
            raise ValueError(f"Unsupported context dropout mode: {self.mode}")
        if not 0.0 <= self.p_clean <= 1.0:
            raise ValueError("p_clean must be in [0.0, 1.0]")
        if not 0.0 <= self.p_full <= 1.0 or not 0.0 <= self.p_block <= 1.0:
            raise ValueError("p_full and p_block must be in [0.0, 1.0]")
        if self.p_full + self.p_block > 1.0 + 1e-9:
            raise ValueError("p_full + p_block must not exceed 1.0")
        if not 0.0 <= self.min_keep <= self.max_keep <= 1.0:
            raise ValueError("min_keep and max_keep must satisfy 0.0 <= min_keep <= max_keep <= 1.0")
        if self.num_blocks_min < 1 or self.num_blocks_min > self.num_blocks_max:
            raise ValueError("num_blocks bounds must satisfy 1 <= min <= max")
        if self.block_h_min < 1 or self.block_h_min > self.block_h_max:
            raise ValueError("block_h bounds must satisfy 1 <= min <= max")
        if self.block_w_min < 1 or self.block_w_min > self.block_w_max:
            raise ValueError("block_w bounds must satisfy 1 <= min <= max")

    def to_dict(self) -> dict[str, object]:
        """Serialize configuration for JSON logging."""
        return asdict(self)


def assert_no_smap_input(inputs: torch.Tensor) -> None:
    """Verify SMAP value and mask channels are strictly zeroed for v4 runs.

    Args:
        inputs: Model input tensor with shape ``[batch, 276, height, width]``.

    Raises:
        ValueError: If input channel count is not 276.
        RuntimeError: If any SMAP value or mask entry is nonzero.
    """
    if inputs.ndim != 4 or inputs.shape[1] != 276:
        raise ValueError(f"Expected input tensor [B, 276, H, W], got {tuple(inputs.shape)}")
    smap_val_max = float(inputs[:, SMAP_VALUE_SLICE].abs().max().item())
    smap_mask_max = float(inputs[:, SMAP_MASK_SLICE].abs().max().item())
    if smap_val_max > 0.0 or smap_mask_max > 0.0:
        raise RuntimeError(
            "Nonzero SMAP channels detected in v4 input! "
            "Ensure PYTHONPATH starts with parallel_training/ablate_site and ABLATE_MODE=no_smap."
        )


def zero_all_context(inputs: torch.Tensor) -> torch.Tensor:
    """Return a cloned input tensor with all Argo C channels zeroed.

    Args:
        inputs: Input tensor with shape ``[batch, 276, height, width]``.

    Returns:
        Cloned tensor where channels ``30:276`` are zeroed and ``0:30`` are preserved.
    """
    output = inputs.clone()
    output[:, ALL_CONTEXT_SLICE] = 0.0
    return output


def _rand_scalar(generator: torch.Generator | None) -> float:
    """Sample a single float in [0, 1) on CPU using the optional generator."""
    return float(torch.rand((), generator=generator, device="cpu").item())


def _randint_scalar(low: int, high_inclusive: int, generator: torch.Generator | None) -> int:
    """Sample an integer in [low, high_inclusive] on CPU using the optional generator."""
    return int(torch.randint(low, high_inclusive + 1, (), generator=generator, device="cpu").item())


def _sample_regime(config: ContextDropoutConfig, generator: torch.Generator | None) -> str:
    """Select the per-step dropout regime according to configuration."""
    if config.mode != "mixed":
        return config.mode
    if _rand_scalar(generator) < config.p_clean:
        return "clean"
    draw = _rand_scalar(generator)
    if draw < config.p_full:
        return "full_drop"
    if draw < config.p_full + config.p_block:
        return "block"
    return "pointwise"


def _build_spatial_keep_mask(
    batch_size: int,
    height: int,
    width: int,
    regime: str,
    config: ContextDropoutConfig,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Construct a CPU boolean keep mask of shape ``[batch, 1, height, width]``."""
    if regime == "full_drop":
        return torch.zeros((batch_size, 1, height, width), dtype=torch.bool, device="cpu")
    if regime == "pointwise":
        keep_prob = config.min_keep + (config.max_keep - config.min_keep) * _rand_scalar(generator)
        return torch.rand((batch_size, 1, height, width), generator=generator, device="cpu") < keep_prob
    if regime == "block":
        keep_mask = torch.ones((batch_size, 1, height, width), dtype=torch.bool, device="cpu")
        for b in range(batch_size):
            num_blocks = _randint_scalar(config.num_blocks_min, config.num_blocks_max, generator)
            for _ in range(num_blocks):
                block_h = min(height, _randint_scalar(config.block_h_min, config.block_h_max, generator))
                block_w = min(width, _randint_scalar(config.block_w_min, config.block_w_max, generator))
                top = _randint_scalar(0, max(0, height - block_h), generator)
                left = _randint_scalar(0, max(0, width - block_w), generator)
                keep_mask[b, 0, top : top + block_h, left : left + block_w] = False
        return keep_mask
    raise ValueError(f"Unexpected dropout regime: {regime}")


def apply_context_dropout(
    inputs: torch.Tensor,
    config: ContextDropoutConfig,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, float, str]:
    """Apply non-mutating Argo C context dropout to a batch of model inputs.

    Surface channels ``0:30`` are preserved unchanged. For every dropped spatial cell
    ``(h, w)``, all Argo C channels ``30:276`` (CT/SA values, CT/SA masks, spatial/time
    offsets, and offset validity masks) are zeroed together so dropped locations leave
    no metadata leakage.

    Args:
        inputs: Input tensor with shape ``[batch, 276, height, width]``.
        config: Context dropout configuration.
        generator: Optional CPU ``torch.Generator`` for deterministic augmentation.

    Returns:
        Tuple of ``(augmented_inputs, kept_valid_fraction, applied_regime)``.
    """
    if inputs.ndim != 4 or inputs.shape[1] != 276:
        raise ValueError(f"Expected input tensor [B, 276, H, W], got {tuple(inputs.shape)}")

    if config.mode == "none":
        return inputs, 1.0, "clean"

    regime = _sample_regime(config, generator)
    if regime == "clean":
        return inputs, 1.0, "clean"

    batch_size, _, height, width = inputs.shape
    keep_bool_cpu = _build_spatial_keep_mask(batch_size, height, width, regime, config, generator)
    keep_bool = keep_bool_cpu.to(device=inputs.device)

    valid_spatial = inputs[:, CONTEXT_MASK_SLICE].amax(dim=1, keepdim=True) > 0.0
    valid_count = int(valid_spatial.sum().item())
    kept_fraction = (
        float((valid_spatial & keep_bool).sum().item()) / float(valid_count) if valid_count > 0 else 1.0
    )

    augmented = inputs.clone()
    if regime == "full_drop":
        augmented[:, ALL_CONTEXT_SLICE] = 0.0
    else:
        augmented[:, ALL_CONTEXT_SLICE] *= keep_bool.to(dtype=augmented.dtype)
    return augmented, kept_fraction, regime
