"""Verify scientific comparison rules using explicit artificial unit-test data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from training_visualization.analysis import mean_std, summarize
from training_visualization.records import COHORTS, SPLIT_MONTHS, SPLITS, load_run


def create_run(
    root: Path,
    seed: int = 42,
    *,
    rmse: float = 1.0,
    tested: bool = True,
    points: int = 100,
    model: str = "residual_cnn",
) -> Path:
    """Build labelled artificial files for unit tests, never research results."""
    root.mkdir(parents=True)
    config = {
        "architecture": model,
        "seed": seed,
        "epochs": 2,
        "width": 48,
        "argo_sampler": "native_coordinates",
        "dataset_manifest_sha256": "unit-test-only",
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    rows = [
        {"epoch": 1, "validation_score": 2.0},
        {"epoch": 2, "validation_score": 1.0},
    ]
    (root / "metrics.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if tested:
        metrics: dict[str, Any] = {}
        for split in SPLITS:
            cohorts = {}
            for cohort in COHORTS:
                cohorts[cohort] = {
                    "profiles_with_valid_levels": 4,
                    "platforms": 2,
                    **{name: {"valid_points": points, "rmse": rmse, "bias": -0.1, "r2": 0.9} for name in ("CT", "SA")},
                }
            metrics[split] = {"months": SPLIT_MONTHS[split], "operators": {"native_coordinates": cohorts}}
        report = {"epoch": 2, "manifest_sha256": "unit-test-only", "metrics": metrics}
        (root / "test_metrics.json").write_text(json.dumps(report), encoding="utf-8")
    return root


def test_seed_summary_uses_sample_sd(tmp_path: Path) -> None:
    """Three seed scores must produce SD with n-1, not standard error."""
    runs = [
        load_run(create_run(tmp_path / str(seed), seed, rmse=float(index)))
        for index, seed in enumerate((42, 43, 44), 1)
    ]
    summary, _, _ = summarize(runs, (42, 43, 44), allow_partial=False)
    assert summary[0].rmse_mean == 2.0
    assert summary[0].rmse_std == 1.0
    assert summary[0].tested_runs == 3


def test_missing_seeds_require_explicit_preview(tmp_path: Path) -> None:
    """An unfinished suite must not be exported as a completed comparison."""
    run = load_run(create_run(tmp_path / "one", 42))
    with pytest.raises(ValueError, match="missing seeds"):
        summarize([run], (42, 43, 44), allow_partial=False)
    rows, _, _ = summarize([run], (42, 43, 44), allow_partial=True)
    assert rows[0].missing_seeds == (43, 44)
    assert rows[0].rmse_std is None


def test_different_observation_support_is_rejected(tmp_path: Path) -> None:
    """The same-looking metric cannot mix different valid-point counts."""
    runs = [load_run(create_run(tmp_path / str(seed), seed, points=count)) for seed, count in ((42, 100), (43, 99))]
    with pytest.raises(ValueError, match="observation support"):
        summarize(runs, (42, 43), allow_partial=False)


def test_different_manifests_are_rejected(tmp_path: Path) -> None:
    """Different normalization/data protocols cannot silently be pooled."""
    first = load_run(create_run(tmp_path / "one", 42))
    second = load_run(create_run(tmp_path / "two", 43))
    second.config["dataset_manifest_sha256"] = "another-test-protocol"
    with pytest.raises(ValueError, match="different dataset manifests"):
        summarize([first, second], (42, 43), allow_partial=False)


def test_final_partial_json_line_can_be_ignored(tmp_path: Path) -> None:
    """A live writer's truncated tail must preserve earlier valid epochs."""
    root = create_run(tmp_path / "one", tested=False)
    with (root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"epoch":3,"validation_score":')
    run = load_run(root, allow_partial=True)
    assert run.completed_epochs == 2
    assert run.notes


def test_malformed_middle_line_is_not_hidden(tmp_path: Path) -> None:
    """Real log corruption must not be interpreted as incomplete training."""
    root = create_run(tmp_path / "one", tested=False)
    (root / "metrics.jsonl").write_text('{"epoch":1}\nBAD\n{"epoch":2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="Malformed JSON log record"):
        load_run(root, allow_partial=True)


def test_test_epoch_must_come_from_validation(tmp_path: Path) -> None:
    """Never accept an arbitrary epoch merely because its test score is better."""
    root = create_run(tmp_path / "one")
    p = root / "test_metrics.json"
    report = json.loads(p.read_text())
    report["epoch"] = 1
    p.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="validation-best epoch"):
        load_run(root)


def test_deterministic_reference_has_no_fabricated_sd(tmp_path: Path) -> None:
    """A single deterministic reference needs no missing random-seed runs."""
    run = load_run(create_run(tmp_path / "climatology", model="climatology_baseline"))
    rows, _, _ = summarize([run], (42, 43, 44), allow_partial=False)
    assert rows[0].expected_runs == 1
    assert rows[0].rmse_std is None
    assert mean_std([None]) == (None, None)


def test_duplicate_seed_runs_are_rejected(tmp_path: Path) -> None:
    """Retraining the same seed twice must not count as two independent seeds."""
    runs = [load_run(create_run(tmp_path / name, 42)) for name in ("original", "retry")]
    with pytest.raises(ValueError, match="Duplicate seeds"):
        summarize(runs, (42,), allow_partial=False)


def test_hyperparameter_variants_are_not_merged(tmp_path: Path) -> None:
    """Different feature widths form separate comparisons, even with one architecture."""
    first = create_run(tmp_path / "narrow", 42)
    second = create_run(tmp_path / "wide", 42)
    path = second / "config.json"
    config = json.loads(path.read_text())
    config["width"] = 96
    path.write_text(json.dumps(config))
    summaries, _, _ = summarize([load_run(first), load_run(second)], (42,), allow_partial=False)
    assert len({row.group for row in summaries}) == 2


def test_live_tail_is_not_a_final_comparison(tmp_path: Path) -> None:
    """A readable old test report cannot make an actively appended log final."""
    root = create_run(tmp_path / "one")
    with (root / "metrics.jsonl").open("a", encoding="utf-8") as stream:
        stream.write('{"epoch":3')
    with pytest.raises(ValueError, match="still being appended"):
        load_run(root)
