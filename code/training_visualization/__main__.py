"""Create read-only training comparisons from existing JSON files."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from training_visualization.analysis import run_inventory, summarize, summary_dicts
from training_visualization.plots import comparison_plots, completion_and_timing, learning_curves
from training_visualization.records import discover_runs


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Export raw physical scores without rounding their underlying values."""
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_report(args: argparse.Namespace) -> Path:
    """Read compatible source artifacts and write a new standalone report directory.

    Input files are opened read-only. No model code, weights, GPU frameworks,
    training processes or raw scientific datasets are accessed.
    """
    output: Path = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    runs = discover_runs(args.runs_root, args.reference_run, allow_partial=args.allow_partial)
    summaries, observations, notes = summarize(runs, tuple(args.expected_seeds), allow_partial=args.allow_partial)
    if not summaries and not args.allow_partial:
        raise ValueError("No completed test scores are available")
    output.mkdir(parents=True)
    formats = tuple(args.formats)
    figures = learning_curves(runs, output, formats, partial=args.allow_partial)
    figures += comparison_plots(summaries, observations, output, formats, partial=args.allow_partial)
    figures += completion_and_timing(runs, output, formats, partial=args.allow_partial)
    inventory = run_inventory(runs)
    write_csv(output / "run_inventory.csv", inventory)
    write_csv(output / "test_scores_per_run.csv", [asdict(row) for row in observations])
    write_csv(output / "test_summary.csv", summary_dicts(summaries))
    payload = {
        "created_utc": datetime.now(UTC).isoformat(),
        "report_mode": "partial_snapshot" if args.allow_partial else "completed_comparison",
        "expected_neural_seeds": args.expected_seeds,
        "manifest_sha256": runs[0].manifest_sha256,
        "runs": inventory,
        "source_hashes": {str(run.path): run.source_hashes for run in runs},
        "summary": summary_dicts(summaries),
        "notes": notes,
        "figures": figures,
    }
    (output / "comparison.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    title = "PARTIAL TRAINING SNAPSHOT" if args.allow_partial else "Training result comparison"
    description = [
        "# " + title,
        "",
        "This report reads only config.json, metrics.jsonl and test_metrics.json.",
        "No training, inference, weight loading, GPU access or source modification was performed.",
        "",
        "- Physical units: CT in deg C; SA in g/kg.",
        "- Means and sample standard deviations summarize independent training seeds, not observations.",
        "- A single deterministic baseline or historical reference has no estimated seed standard deviation.",
        "- Validation-selected checkpoints are used; test results do not select models or epochs.",
        "- CSV files retain unrounded values. comparison.json records source hashes and all warnings.",
        "- Spatial maps, scatterplots and depth-wise metrics require predictions/observations, which these JSON files do not contain.",
        "",
        "## Notes",
        "",
        *["- " + note for note in notes],
        "",
        "## Figures",
        "",
        *[f"- [{name}]({name})" for name in figures],
        "",
    ]
    (output / "README.md").write_text("\n".join(description), encoding="utf-8")
    return output


def main() -> None:
    """Parse explicit source directories and export figures to a fresh location."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, nargs="+", required=True, help="Explicit suite/model directories")
    parser.add_argument(
        "--reference-run", type=Path, action="append", default=[], help="Optional singleton historical reference"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-seeds", type=int, nargs="+", default=[20261117, 20261118, 20261119])
    parser.add_argument("--formats", nargs="+", choices=("png", "pdf", "svg"), default=["png", "pdf"])
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Label incomplete live snapshots; never fill missing scores with zero",
    )
    args = parser.parse_args()
    try:
        output = build_report(args)
    except (ValueError, FileNotFoundError, FileExistsError) as error:
        parser.exit(2, f"Visualization could not be created: {error}\n")
    print(f"Visualization report: {output}")


if __name__ == "__main__":
    main()
