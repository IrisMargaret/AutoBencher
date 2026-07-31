"""Aggregate completed study results across seeds."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

from autobencher.experiment import atomic_json
from autobencher.result_schema import ExperimentRecord, validate_registry
from autobencher.statistics import aggregate_records


def _latest_summary(experiment_dir: Path) -> Path | None:
    candidates = list(experiment_dir.rglob("experiment_summary.json"))
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime_ns)


def collect_results(index_path: Path) -> list[dict[str, Any]]:
    with index_path.open("r", encoding="utf-8") as handle:
        registry = json.load(handle)
    validate_registry(registry)
    results = []
    for payload in registry["experiments"]:
        record = ExperimentRecord.from_dict(payload)
        summary_path = _latest_summary(Path(record.experiment_dir))
        summary = {}
        if summary_path:
            with summary_path.open("r", encoding="utf-8") as handle:
                summary = json.load(handle)
        results.append(
            {
                "study_id": record.study_id,
                "method": record.method,
                "variant": record.variant,
                "seed": record.seed,
                "model": record.model,
                "budget": record.budget,
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
        "--metric",
        choices=("baseline_accuracy", "final_accuracy", "accuracy_delta"),
        default="final_accuracy",
    )
    args = parser.parse_args()
    index_path = Path(args.index).expanduser().resolve()
    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output
        else index_path.parent / "aggregate.json"
    )
    records = collect_results(index_path)
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
