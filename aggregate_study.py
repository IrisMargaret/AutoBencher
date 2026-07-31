"""Aggregate completed study results across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

from autobencher.experiment import atomic_json  # noqa: E402
from autobencher.paper_results import build_paper_tables  # noqa: E402
from autobencher.result_schema import (  # noqa: E402
    ExperimentRecord,
    validate_registry,
)
from autobencher.statistics import aggregate_records  # noqa: E402
from autobencher.study_runner import (  # noqa: E402
    StudyConfigurationError,
    validate_experiment_completion,
)


def collect_results(
    index_path: Path,
    *,
    allow_partial_development_results: bool = False,
) -> list[dict[str, Any]]:
    with index_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    validate_registry(registry)
    results = []
    for payload in registry["experiments"]:
        record = ExperimentRecord.from_dict(payload)
        if record.status != "completed":
            if allow_partial_development_results:
                continue
            raise StudyConfigurationError(
                f"Experiment is not completed: {record.study_id}"
            )
        try:
            completion = validate_experiment_completion(record)
        except StudyConfigurationError:
            if allow_partial_development_results:
                continue
            raise
        summary_path = Path(record.summary_path).resolve()
        summary = {}
        if summary_path:
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        results.append(
            {
                "study_id": record.study_id,
                "method": record.method,
                "variant": record.variant,
                "budget_protocol": record.budget_protocol,
                "seed": record.seed,
                "model": record.model,
                "budget": record.budget,
                "evaluation_set_id": completion["evaluation_set_id"],
                "evaluation_set_version": completion[
                    "evaluation_set_version"
                ],
                "evaluation_set_sha256": completion[
                    "evaluation_set_sha256"
                ],
                "checkpoint_sha256": completion["checkpoint_sha256"],
                "status": record.status,
                "baseline_accuracy": summary.get("baseline_accuracy"),
                "final_accuracy": summary.get("final_accuracy"),
                "accuracy_delta": summary.get("accuracy_delta"),
                "summary_path": str(summary_path) if summary_path else None,
            }
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Aggregate a Study Runner experiment index."
    )
    parser.add_argument("--index", required=True)
    parser.add_argument("--output")
    parser.add_argument(
        "--results-dir",
        help="Directory for paper-ready CSV tables; defaults to INDEX_DIR/results.",
    )
    parser.add_argument(
        "--metric",
        choices=("baseline_accuracy", "final_accuracy", "accuracy_delta"),
        default="final_accuracy",
    )
    parser.add_argument(
        "--allow-partial-development-results",
        action="store_true",
        help="Development only: omit incomplete cells instead of failing closed.",
    )
    args = parser.parse_args()
    index_path = Path(args.index).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else index_path.parent / "aggregate.json"
    )
    records = collect_results(
        index_path,
        allow_partial_development_results=(
            args.allow_partial_development_results
        ),
    )
    payload = {
        "schema_version": "1.0",
        "experiment_index": str(index_path),
        "metric": args.metric,
        "per_experiment": records,
        "groups": aggregate_records(records, metric=args.metric),
    }
    atomic_json(payload, output_path)
    csv_path = output_path.with_suffix(".csv")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = list(records[0]) if records else [
            "study_id",
            "method",
            "variant",
            "seed",
            "model",
            "budget",
            "status",
            "baseline_accuracy",
            "final_accuracy",
            "accuracy_delta",
            "summary_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    print(f"Aggregate JSON: {output_path}")
    print(f"Per-experiment CSV: {csv_path}")
    results_dir = (
        Path(args.results_dir).expanduser().resolve()
        if args.results_dir
        else index_path.parent / "results"
    )
    paper = build_paper_tables(
        index_path,
        results_dir,
        allow_partial_development_results=(
            args.allow_partial_development_results
        ),
    )
    print(f"Paper tables: {paper['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
