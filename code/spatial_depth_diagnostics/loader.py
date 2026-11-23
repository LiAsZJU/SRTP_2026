"""Unified read-only checkpoint, v2 dataset, and native-grid inference helpers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from netCDF4 import Dataset

DevicePreference = Literal["auto", "cpu", "cuda"]

MODEL_LABELS: dict[str, str] = {
    "climatology_baseline": "Climatology",
    "eof_ridge": "EOF + ridge",
    "residual_cnn": "Residual CNN",
    "unet_baseline": "U-Net",
    "vit_baseline": "ViT",
    "swin": "Swin",
}


@dataclass(frozen=True)
class GridAxes:
    """Native rectilinear coordinate axes of the Tropical Pacific v2 grid."""

    depth: np.ndarray
    latitude: np.ndarray
    longitude: np.ndarray


@dataclass(frozen=True)
class ModelSpec:
    """User-supplied model label and checkpoint path."""

    label: str
    checkpoint: Path


@dataclass(frozen=True)
class LoadedModel:
    """Immutable inference wrapper around a frozen checkpoint."""

    label: str
    architecture: str
    checkpoint: Path
    checkpoint_sha256: str
    epoch: int
    seed: int | None
    amp: bool
    config: dict[str, Any]
    module: Any
    device: Any


@dataclass(frozen=True)
class MonthInference:
    """Physical gridded and Argo-profile reconstructions for one target month."""

    month: str
    split: str
    predicted_grid: np.ndarray
    background_grid: np.ndarray
    cora_grid: np.ndarray
    ocean_mask: np.ndarray
    cora_mask: np.ndarray
    profile_observed: np.ndarray
    profile_predicted: np.ndarray
    profile_background: np.ndarray
    profile_valid: np.ndarray
    profile_lat: np.ndarray
    profile_lon: np.ndarray
    profile_time_offset: np.ndarray
    profile_role_code: np.ndarray
    profile_platform: np.ndarray


def parse_model_spec(raw: str) -> ModelSpec:
    """Parse ``label=/path/to/best.pt`` or infer a label from the checkpoint path.

    Args:
        raw: Either ``label=path`` or a plain filesystem path.

    Returns:
        Parsed ModelSpec with a non-empty label and checkpoint Path.

    Raises:
        ValueError: If the specification is empty or lacks a valid path.
    """
    text = raw.strip()
    if not text:
        raise ValueError("Empty model specification")
    if "=" in text:
        label, path_str = text.split("=", 1)
        label = label.strip()
        path = Path(path_str.strip())
    else:
        path = Path(text)
        label = path.parent.name or path.stem
    if not label:
        raise ValueError(f"Empty model label in specification: {raw!r}")
    if not str(path):
        raise ValueError(f"Empty checkpoint path in specification: {raw!r}")
    return ModelSpec(label=label, checkpoint=path)


def file_sha256(path: Path) -> str:
    """Compute SHA-256 hex digest of a file using streamed reads."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_device(preference: DevicePreference, cpu_threads: int) -> Any:
    """Resolve PyTorch execution device without silently falling back on explicit cuda.

    Args:
        preference: Requested device mode (``auto``, ``cpu``, or ``cuda``).
        cpu_threads: Number of intra-op CPU threads to configure.

    Returns:
        Resolved ``torch.device``.

    Raises:
        RuntimeError: If ``cuda`` is explicitly requested without an available GPU.
    """
    import torch

    if cpu_threads < 1:
        raise ValueError(f"cpu_threads must be >= 1, got {cpu_threads}")
    torch.set_num_threads(cpu_threads)
    if preference == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was explicitly requested, but no GPU is available")
        return torch.device("cuda")
    if preference == "auto" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_grid_axes(background_path: Path) -> GridAxes:
    """Load validated native depth, latitude, and longitude axes from NetCDF."""
    with Dataset(background_path) as dataset:
        depth = np.asarray(dataset["depth"][:], dtype=np.float64)
        latitude = np.asarray(dataset["latitude"][:], dtype=np.float64)
        longitude = np.asarray(dataset["longitude"][:], dtype=np.float64)
    for name, axis in (("depth", depth), ("latitude", latitude), ("longitude", longitude)):
        if axis.ndim != 1 or axis.size < 2 or not np.all(np.diff(axis) > 0) or not np.isfinite(axis).all():
            raise ValueError(f"Invalid {name} axis in {background_path}")
    return GridAxes(depth=depth, latitude=latitude, longitude=longitude)


def infer_architecture(config: dict[str, Any]) -> str:
    """Identify which of the six supported v2 architectures a checkpoint stores."""
    schema = config.get("schema")
    if schema == "ocean_baseline_checkpoint_v1":
        architecture = str(config["architecture"])
        if architecture not in MODEL_LABELS:
            raise ValueError(f"Unsupported baseline architecture: {architecture}")
        return architecture
    if all(key in config for key in ("input_channels", "dim", "blocks", "heads", "window")):
        return "swin"
    raise ValueError(f"Unrecognized checkpoint config schema: {sorted(config)}")


def instantiate_model(config: dict[str, Any]) -> tuple[str, Any]:
    """Instantiate a frozen v2 architecture from its checkpoint configuration."""
    architecture = infer_architecture(config)
    if architecture == "swin":
        from train_swin_physics import SwinReconstructor

        module = SwinReconstructor(
            int(config["input_channels"]),
            int(config["dim"]),
            int(config["blocks"]),
            int(config["heads"]),
            int(config["window"]),
        )
        return architecture, module
    if architecture == "climatology_baseline":
        from climatology_baseline.model import ClimatologyBaseline

        return architecture, ClimatologyBaseline()
    if architecture == "eof_ridge":
        from eof_ridge.model import EOFRidge

        return architecture, EOFRidge(int(config["modes"]))
    if architecture == "residual_cnn":
        from residual_cnn.model import ResidualCNN

        return architecture, ResidualCNN(int(config["width"]), int(config["blocks"]))
    if architecture == "unet_baseline":
        from unet_baseline.model import UNetReconstructor

        return architecture, UNetReconstructor(int(config["width"]))
    if architecture == "vit_baseline":
        from vit_baseline.model import ViTReconstructor

        return architecture, ViTReconstructor(
            int(config["embedding_dim"]),
            int(config["blocks"]),
            int(config["heads"]),
            float(config["dropout"]),
        )
    raise ValueError(f"Unsupported architecture: {architecture}")


def load_frozen_model(
    spec: ModelSpec,
    manifest_sha256: str,
    device: Any,
) -> LoadedModel:
    """Load an immutable checkpoint and verify its v2 dataset manifest hash.

    Args:
        spec: Label and checkpoint path.
        manifest_sha256: Expected SHA-256 hex digest of ``training_manifest.json``.
        device: Target ``torch.device``.

    Returns:
        LoadedModel in eval mode.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If the checkpoint manifest hash does not match the dataset.
    """
    import torch

    checkpoint_path = spec.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    config = dict(checkpoint["config"])
    expected_hash = config.get("dataset_manifest_sha256")
    if expected_hash is not None and str(expected_hash) != manifest_sha256:
        raise ValueError(
            f"Manifest hash mismatch for {spec.label}: checkpoint expects {expected_hash}, "
            f"dataset has {manifest_sha256}"
        )
    architecture, module = instantiate_model(config)
    module.load_state_dict(checkpoint["model"])
    module = module.to(device)
    module.eval()
    raw_seed = config.get("seed")
    seed = int(raw_seed) if raw_seed is not None else None
    return LoadedModel(
        label=spec.label,
        architecture=architecture,
        checkpoint=checkpoint_path,
        checkpoint_sha256=file_sha256(checkpoint_path),
        epoch=int(checkpoint.get("epoch", 0)),
        seed=seed,
        amp=bool(config.get("amp", False)),
        config=config,
        module=module,
        device=device,
    )


def target_norm_stats(source: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mean, std)`` arrays of shape ``(2, 60)`` for CT and SA anomalies."""
    ct_stats = source.statistics["target.CT_anomaly"]
    sa_stats = source.statistics["target.SA_anomaly"]
    mean = np.stack(
        (np.asarray(ct_stats["mean"], dtype=np.float32), np.asarray(sa_stats["mean"], dtype=np.float32)),
        axis=0,
    )
    std = np.stack(
        (np.asarray(ct_stats["std"], dtype=np.float32), np.asarray(sa_stats["std"], dtype=np.float32)),
        axis=0,
    )
    return mean, std


def open_v2_dataset(dataset_dir: Path) -> tuple[Any, str, GridAxes, set[str]]:
    """Open v2 TrainingDataset, verify manifest, load native axes and validation platforms."""
    from dataset_v2.evaluate_checkpoint import eligible_validation_platforms
    from server_training_data import TrainingDataset

    manifest_path = dataset_dir / "training_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing dataset manifest: {manifest_path}")
    manifest_sha256 = file_sha256(manifest_path)
    source = TrainingDataset(dataset_dir)
    first_background = Path(next(iter(source.records.values()))["background"])
    axes = load_grid_axes(first_background)
    validation_platforms = eligible_validation_platforms(source)
    return source, manifest_sha256, axes, validation_platforms


def resolve_months(
    source: Any,
    splits: list[str] | None = None,
    explicit_months: list[str] | None = None,
) -> list[tuple[str, str]]:
    """Resolve ordered ``(month, split)`` pairs from split names or explicit months."""
    if explicit_months:
        pairs: list[tuple[str, str]] = []
        for month in explicit_months:
            if month not in source.records:
                raise ValueError(f"Unknown month {month!r} in dataset manifest")
            pairs.append((month, str(source.records[month]["split"])))
        return pairs
    if not splits:
        raise ValueError("Specify either --splits or --months")
    resolved: list[tuple[str, str]] = []
    for split in splits:
        split_months = sorted(month for month, row in source.records.items() if row["split"] == split)
        if not split_months:
            raise ValueError(f"Empty or unknown split {split!r}")
        resolved.extend((month, split) for month in split_months)
    return resolved


def run_month_inference(
    source: Any,
    month: str,
    loaded: LoadedModel | None,
    sampler: Any | None,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
    direct_source: Literal["cora", "background"] | None = None,
) -> MonthInference:
    """Reconstruct physical 3D fields and native-sampled Argo profiles for one month.

    Args:
        source: Open ``TrainingDataset`` instance.
        month: Six-digit month identifier (e.g., ``"202401"``).
        loaded: LoadedModel for checkpoint inference, or ``None`` for direct data slicing.
        sampler: Native ``GridSampler`` aligned with ``loaded.device`` (required if ``loaded`` is set).
        norm_mean: Target anomaly normalization mean, shape ``(2, 60)``.
        norm_std: Target anomaly normalization std, shape ``(2, 60)``.
        direct_source: When ``loaded`` is ``None``, choose ``"cora"`` or ``"background"``.

    Returns:
        MonthInference containing physical CT/SA grids (land masked to NaN) and profile arrays.
    """
    import torch
    from dataset_v2.evaluate_checkpoint import load_profiles
    from train_swin_physics import model_input

    sample = source.load(month)
    roles, platforms = load_profiles(source.records[month]["argo_profiles"])
    background_mask = np.asarray(sample["background_mask"], dtype=bool)
    cora_mask = np.asarray(sample["target_mask"], dtype=bool)
    background_grid = np.where(background_mask, np.asarray(sample["background"], dtype=np.float32), np.nan)
    cora_grid = np.where(cora_mask, np.asarray(sample["target_absolute"], dtype=np.float32), np.nan)

    profile_observed = np.asarray(sample["profile_absolute"], dtype=np.float32)
    profile_background = np.asarray(sample["profile_background"], dtype=np.float32)
    profile_valid = np.asarray(sample["profile_mask"], dtype=bool)
    positions_np = np.asarray(sample["profile_position"], dtype=np.float32)

    if loaded is None:
        if direct_source == "cora":
            predicted_grid = cora_grid.copy()
        elif direct_source == "background":
            predicted_grid = background_grid.copy()
        else:
            raise ValueError(f"direct_source must be 'cora' or 'background' when loaded is None, got {direct_source!r}")
        profile_predicted = profile_background.copy()
    else:
        if sampler is None:
            raise ValueError("GridSampler is required when running checkpoint inference")
        with torch.no_grad():
            inputs = model_input(sample, loaded.device)
            use_amp = loaded.device.type == "cuda" and loaded.amp
            with torch.autocast(device_type=loaded.device.type, dtype=torch.float16, enabled=use_amp):
                pred_norm_grid = cast(torch.Tensor, loaded.module(inputs))
            pred_norm_grid_f32 = pred_norm_grid.float()
            positions_t = torch.from_numpy(positions_np).to(loaded.device)
            pred_norm_profiles = sampler(pred_norm_grid_f32, positions_t).cpu().numpy()
            pred_norm_grid_np = pred_norm_grid_f32[0].cpu().numpy()

        predicted_grid = np.where(
            background_mask,
            background_grid + pred_norm_grid_np * norm_std[:, :, None, None] + norm_mean[:, :, None, None],
            np.nan,
        ).astype(np.float32)
        profile_predicted = (
            profile_background + pred_norm_profiles * norm_std[None, :, :] + norm_mean[None, :, :]
        ).astype(np.float32)

    return MonthInference(
        month=month,
        split=str(sample["split"]),
        predicted_grid=predicted_grid,
        background_grid=background_grid.astype(np.float32),
        cora_grid=cora_grid.astype(np.float32),
        ocean_mask=background_mask,
        cora_mask=cora_mask,
        profile_observed=profile_observed,
        profile_predicted=profile_predicted,
        profile_background=profile_background,
        profile_valid=profile_valid,
        profile_lat=positions_np[:, 0],
        profile_lon=positions_np[:, 1],
        profile_time_offset=positions_np[:, 2],
        profile_role_code=np.asarray(roles, dtype=np.int64),
        profile_platform=np.asarray(platforms, dtype=str),
    )
