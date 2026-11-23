"""Read existing JSON artifacts without importing or executing training code."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODEL_NAMES = {
    "climatology_baseline": "Climatology",
    "eof_ridge": "EOF + ridge",
    "residual_cnn": "Residual CNN",
    "unet_baseline": "U-Net",
    "vit_baseline": "ViT",
    "swin": "Swin",
}
DETERMINISTIC_MODELS = frozenset(("climatology_baseline", "eof_ridge"))
SPLITS = ("test_locked", "forward_test")
SPLIT_MONTHS = {
    "test_locked": tuple(f"2024{month:02d}" for month in range(1, 13)),
    "forward_test": tuple(f"2025{month:02d}" for month in range(1, 7)),
}
COHORTS = ("all_E", "E_platforms_unseen_in_validation")
VARIABLES = ("CT", "SA")
VARIANT_FIELDS = (
    "epochs",
    "patience",
    "lr",
    "weight_decay",
    "grad_clip",
    "lambda_query",
    "lambda_physics",
    "stability_scale",
    "stability_tolerance",
    "amp",
    "input_channels",
    "output_order",
    "output_shape",
    "argo_sampler",
    "train_months",
    "validation_months",
    "width",
    "blocks",
    "dim",
    "heads",
    "window",
    "embedding_dim",
    "dropout",
    "patch_size",
    "modes",
    "ridge_alpha",
    "target_missing_rule",
)


def finite_number(value: object, location: str) -> float | None:
    """Accept a finite JSON number or an explicit missing value."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"Expected a finite number or null at {location}")
    return float(value)


@dataclass(frozen=True)
class TestRow:
    """One physical-variable score on a fixed split and Argo platform cohort."""

    run: str
    model: str
    group: str
    seed: int | None
    split: str
    cohort: str
    variable: str
    months: tuple[str, ...]
    valid_points: int
    profiles: int
    platforms: int
    rmse: float | None
    bias: float | None
    r2: float | None


@dataclass
class RunRecord:
    """A single run's configuration, observed log snapshot and optional test report."""

    path: Path
    config: dict[str, Any]
    history: list[dict[str, Any]]
    report: dict[str, Any] | None
    model: str
    group: str
    seed: int | None
    is_reference: bool
    source_hashes: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    @property
    def planned_epochs(self) -> int:
        """Return the configured upper limit, not an inferred completed count."""
        return int(self.config.get("epochs", 0))

    @property
    def completed_epochs(self) -> int:
        """Return the greatest logged epoch; climatology's packaging epoch is zero."""
        return max((int(row["epoch"]) for row in self.history), default=0)

    @property
    def best_epoch(self) -> int | None:
        """Select the earliest minimum validation score, never the minimum test error."""
        eligible = [row for row in self.history if row.get("validation_score") is not None]
        return int(min(eligible, key=lambda row: float(row["validation_score"]))["epoch"]) if eligible else None

    @property
    def manifest_sha256(self) -> str:
        """Return the provenance key required for cross-model comparisons."""
        return str(self.config["dataset_manifest_sha256"])

    @property
    def display_name(self) -> str:
        """Return the human-readable architecture name."""
        return MODEL_NAMES.get(self.model, self.model)


def _read_bytes(path: Path, hashes: dict[str, str]) -> bytes:
    data = path.read_bytes()
    hashes[str(path.resolve())] = hashlib.sha256(data).hexdigest()
    return data


def _read_history(path: Path, hashes: dict[str, str], notes: list[str]) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = _read_bytes(path, hashes)
    lines = raw.splitlines(keepends=True)
    rows: list[dict[str, Any]] = []
    for index, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index == len(lines) - 1 and not raw.endswith(b"\n"):
                notes.append(f"Ignored an unterminated, incomplete final log line in {path}")
                continue
            raise ValueError(f"Malformed JSON log record: {path}:{index + 1}") from None
        epoch = row.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError(f"Invalid epoch in {path}:{index + 1}")
        if rows and epoch <= rows[-1]["epoch"]:
            raise ValueError(f"Duplicate or unordered epochs in {path}")
        for key, value in row.items():
            if key.startswith(("train_", "validation_")) or key == "seconds":
                finite_number(value, f"{path}:{index + 1}:{key}")
        rows.append(row)
    return rows


def load_run(path: Path, *, is_reference: bool = False, allow_partial: bool = False) -> RunRecord:
    """Read one run and reject incompatible provenance or checkpoint selection.

    Args:
        path: Directory containing config.json and metrics.jsonl.
        is_reference: Treat this explicitly selected historical run as a singleton.
        allow_partial: Permit a still-being-written test JSON as a pending report.

    Returns:
        A snapshot of existing metadata, without opening any weights or NetCDF.
    """
    path = path.resolve()
    hashes: dict[str, str] = {}
    notes: list[str] = []
    config = json.loads(_read_bytes(path / "config.json", hashes))
    if not isinstance(config, dict) or not config.get("dataset_manifest_sha256"):
        raise ValueError(f"Missing manifest SHA256 in {path / 'config.json'}; select compatible v2 runs")
    if config.get("argo_sampler") != "native_coordinates":
        raise ValueError(f"Native-coordinate training sampler is not recorded in {path}")
    model = config.get("architecture")
    if model is None and all(key in config for key in ("dim", "heads", "window")):
        model = "swin"
    if not isinstance(model, str) or model not in MODEL_NAMES:
        raise ValueError(f"Cannot identify the model in {path}")
    signature = {key: config[key] for key in VARIANT_FIELDS if key in config}
    digest = hashlib.sha256(json.dumps(signature, sort_keys=True, allow_nan=False).encode()).hexdigest()[:8]
    group = f"{model}_{digest}" + ("_reference" if is_reference else "")
    history = _read_history(path / "metrics.jsonl", hashes, notes)
    if notes and not allow_partial:
        raise ValueError(f"The log is still being appended in {path}; use --allow-partial for a preview")
    report = None
    report_path = path / "test_metrics.json"
    if report_path.exists():
        try:
            report = json.loads(_read_bytes(report_path, hashes))
        except json.JSONDecodeError:
            if not allow_partial:
                raise ValueError(f"Incomplete or malformed test report: {report_path}") from None
            notes.append(f"Test JSON is not yet readable: {report_path}")
    seed = config.get("seed")
    if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool)):
        raise ValueError(f"Invalid seed in {path}")
    run = RunRecord(path, config, history, report, model, group, seed, is_reference, hashes, notes)
    if report is not None:
        if report.get("manifest_sha256") != run.manifest_sha256:
            raise ValueError(f"Training and test manifests differ in {path}")
        if run.best_epoch is None or report.get("epoch") != run.best_epoch:
            raise ValueError(f"Test report does not use the logged validation-best epoch in {path}")
    return run


def discover_runs(roots: list[Path], references: list[Path], *, allow_partial: bool) -> list[RunRecord]:
    """Discover explicitly selected runs, excluding recoverable archives."""
    paths: dict[Path, bool] = {}
    for root in roots:
        if not root.is_dir():
            raise FileNotFoundError(root)
        for config in root.rglob("config.json"):
            if "_archive" not in config.relative_to(root).parts:
                paths[config.parent.resolve()] = False
    for reference in references:
        paths[reference.resolve()] = True
    if not paths:
        raise ValueError("No config.json files were found under the selected run roots")
    return [
        load_run(path, is_reference=reference, allow_partial=allow_partial) for path, reference in sorted(paths.items())
    ]


def test_rows(run: RunRecord) -> list[TestRow]:
    """Extract native-coordinate physical scores for both periods and both cohorts."""
    if run.report is None:
        return []
    rows = []
    try:
        for split in SPLITS:
            payload = run.report["metrics"][split]
            months = tuple(str(month) for month in payload["months"])
            if months != SPLIT_MONTHS[split]:
                raise ValueError(f"Unexpected test months for {split} in {run.path}")
            for cohort in COHORTS:
                scores = payload["operators"]["native_coordinates"][cohort]
                for variable in VARIABLES:
                    values = scores[variable]
                    count = int(values["valid_points"])
                    rmse = finite_number(values["rmse"], f"{run.path}:{split}:{cohort}:{variable}:rmse")
                    bias = finite_number(values["bias"], f"{run.path}:{split}:{cohort}:{variable}:bias")
                    r2 = finite_number(values["r2"], f"{run.path}:{split}:{cohort}:{variable}:r2")
                    if count < 0 or (count > 0 and (rmse is None or bias is None)) or (rmse is not None and rmse < 0):
                        raise ValueError(f"Invalid test support or error statistics in {run.path}")
                    rows.append(
                        TestRow(
                            str(run.path),
                            run.model,
                            run.group,
                            run.seed,
                            split,
                            cohort,
                            variable,
                            months,
                            count,
                            int(scores["profiles_with_valid_levels"]),
                            int(scores["platforms"]),
                            rmse,
                            bias,
                            r2,
                        )
                    )
    except KeyError as error:
        raise ValueError(f"Required test split/cohort/metric missing in {run.path}: {error}") from error
    return rows
