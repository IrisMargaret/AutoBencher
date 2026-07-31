"""Rebuild deterministic, paper-ready tables from raw study artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from autobencher.result_schema import ExperimentRecord, validate_registry
from autobencher.statistics import (
    holm_correction,
    mcnemar_test,
    paired_bootstrap,
    stratified_seed_item_bootstrap,
    summarize_values,
)


LONG_FIELDS = (
    "study_id",
    "method",
    "variant",
    "budget_protocol",
    "seed",
    "model",
    "cycle",
    "question_id",
    "category",
    "subcategory",
    "difficulty",
    "gold_answer",
    "predicted_answer",
    "is_correct",
    "error_type",
    "confidence",
    "latency",
    "input_tokens",
    "output_tokens",
)


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _latest_run_dir(experiment_dir: Path) -> Path | None:
    candidates = [
        path.parent
        for path in experiment_dir.rglob("experiment_summary.json")
    ]
    if not candidates:
        candidates = [
            path.parent
            for path in experiment_dir.rglob("run_manifest.json")
        ]
    return max(candidates, key=lambda path: path.stat().st_mtime_ns) if candidates else None


def _cycle_from_stage(name: str) -> int:
    if name == "baseline":
        return 0
    if name.startswith("cycle_"):
        try:
            return int(name.split("_", 1)[1])
        except ValueError:
            return -1
    return -1


def _confidence(record: Mapping[str, Any]) -> float | None:
    parsed = record.get("parsed_response")
    if isinstance(parsed, Mapping):
        try:
            return float(parsed.get("confidence"))
        except (TypeError, ValueError):
            return None
    return None


def load_results_long(index_path: str | Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index_path = Path(index_path)
    registry = _read_json(index_path)
    validate_registry(registry)
    rows = []
    run_rows = []
    for payload in registry["experiments"]:
        record = ExperimentRecord.from_dict(payload)
        run_dir = _latest_run_dir(Path(record.experiment_dir))
        summary = {}
        ledger = {}
        if run_dir:
            summary_path = run_dir / "experiment_summary.json"
            ledger_path = run_dir / "budget_ledger.json"
            if summary_path.is_file():
                summary = _read_json(summary_path)
            if ledger_path.is_file():
                ledger = _read_json(ledger_path)
            for comparison_path in sorted(
                (run_dir / "fixed_test").glob(
                    "*/fixed_math.compare_answers.json"
                )
            ):
                cycle = _cycle_from_stage(comparison_path.parent.name)
                if cycle < 0:
                    continue
                comparison = _read_json(comparison_path)
                for item in comparison if isinstance(comparison, list) else []:
                    rows.append(
                        {
                            "study_id": record.study_id,
                            "method": record.method,
                            "variant": record.variant,
                            "budget_protocol": record.budget_protocol,
                            "seed": record.seed,
                            "model": record.model,
                            "cycle": cycle,
                            "question_id": item.get(
                                "question_id",
                                item.get("id"),
                            ),
                            "category": item.get("category"),
                            "subcategory": item.get(
                                "sub_category",
                                item.get("subcategory"),
                            ),
                            "difficulty": item.get("difficulty"),
                            "gold_answer": item.get(
                                "gold_answer",
                                item.get("canonical_answer"),
                            ),
                            "predicted_answer": item.get(
                                "test_taker_response"
                            ),
                            "is_correct": bool(item.get("is_correct")),
                            "error_type": item.get(
                                "primary_error_tag",
                                (
                                    item.get("error_tags", [None])[0]
                                    if item.get("error_tags")
                                    else None
                                ),
                            ),
                            "confidence": _confidence(item),
                            "latency": item.get("latency"),
                            "input_tokens": item.get("input_tokens"),
                            "output_tokens": item.get("output_tokens"),
                        }
                    )
        training = ledger.get("training", {})
        efficiency = ledger.get("efficiency", {})
        totals = ledger.get("totals", {})
        retention_final = (
            summary.get("retention_test", {}).get("final", {})
            if isinstance(summary.get("retention_test"), Mapping)
            else {}
        )
        retention_forgetting = (
            retention_final.get("forgetting", {})
            if isinstance(retention_final, Mapping)
            else {}
        )
        run_rows.append(
            {
                "study_id": record.study_id,
                "method": record.method,
                "variant": record.variant,
                "budget_protocol": record.budget_protocol,
                "seed": record.seed,
                "model": record.model,
                "budget": record.budget,
                "status": record.status,
                "baseline_accuracy": summary.get("baseline_accuracy"),
                "final_accuracy": summary.get("final_accuracy"),
                "accuracy_delta": summary.get("accuracy_delta"),
                "retention_accuracy_delta": retention_forgetting.get(
                    "accuracy_delta"
                ),
                "retention_forgetting_rate": retention_forgetting.get(
                    "forgetting_rate"
                ),
                "retention_forgotten_count": retention_forgetting.get(
                    "forgotten_count"
                ),
                "training_sample_count": training.get(
                    "final_training_sample_count"
                ),
                "training_token_count": training.get(
                    "final_training_token_count"
                ),
                "gpu_hours": training.get("gpu_hours"),
                "peak_gpu_memory_gb": training.get("peak_gpu_memory_gb"),
                "total_tokens": totals.get("total_tokens"),
                "total_api_calls": totals.get("total_api_calls"),
                "estimated_cost": ledger.get("estimated_cost", {}).get(
                    "amount"
                ),
                "accepted_samples_per_1k_generated_tokens": efficiency.get(
                    "accepted_samples_per_1k_generated_tokens"
                ),
                "accuracy_gain_per_1k_training_tokens": efficiency.get(
                    "accuracy_gain_per_1k_training_tokens"
                ),
                "accuracy_gain_per_gpu_hour": efficiency.get(
                    "accuracy_gain_per_gpu_hour"
                ),
            }
        )
    rows.sort(
        key=lambda item: (
            item["study_id"],
            item["cycle"],
            str(item["question_id"]),
        )
    )
    run_rows.sort(key=lambda item: item["study_id"])
    return rows, run_rows


def _final_rows(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["study_id"])].append(row)
    output = []
    for values in grouped.values():
        final_cycle = max(int(value["cycle"]) for value in values)
        output.extend(
            dict(value)
            for value in values
            if int(value["cycle"]) == final_cycle
        )
    return output


def _accuracy_rows(
    rows: Iterable[Mapping[str, Any]],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[bool]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row.get(field) for field in group_fields)].append(
            bool(row["is_correct"])
        )
    return [
        {
            **dict(zip(group_fields, key)),
            "question_count": len(values),
            "accuracy": sum(values) / len(values),
        }
        for key, values in sorted(grouped.items(), key=lambda item: str(item[0]))
    ]


def _run_item_metrics(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["study_id"]].append(row)
    output = {}
    for study_id, values in grouped.items():
        by_cycle: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for value in values:
            by_cycle[int(value["cycle"])].append(value)
        accuracies = [
            (cycle, sum(item["is_correct"] for item in items) / len(items))
            for cycle, items in sorted(by_cycle.items())
        ]
        if len(accuracies) > 1:
            auc = sum(
                (left[1] + right[1]) / 2 * (right[0] - left[0])
                for left, right in zip(accuracies, accuracies[1:])
            ) / max(1, accuracies[-1][0] - accuracies[0][0])
        else:
            auc = accuracies[0][1] if accuracies else None
        baseline = {
            str(item["question_id"]): bool(item["is_correct"])
            for item in by_cycle.get(0, [])
        }
        final_cycle = max(by_cycle) if by_cycle else 0
        final = {
            str(item["question_id"]): bool(item["is_correct"])
            for item in by_cycle.get(final_cycle, [])
        }
        initially_correct = [key for key, value in baseline.items() if value]
        forgetting = (
            sum(not final.get(key, False) for key in initially_correct)
            / len(initially_correct)
            if initially_correct
            else None
        )
        final_subcategories = _accuracy_rows(
            by_cycle.get(final_cycle, []),
            ("subcategory",),
        )
        output[study_id] = {
            "learning_curve_auc": auc,
            "forgetting_rate": forgetting,
            "macro_accuracy": (
                sum(item["accuracy"] for item in final_subcategories)
                / len(final_subcategories)
                if final_subcategories
                else None
            ),
            "worst_subcategory_accuracy": (
                min(item["accuracy"] for item in final_subcategories)
                if final_subcategories
                else None
            ),
        }
    return output


def _significance(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    final = _final_rows(rows)
    by_cell: dict[tuple[Any, ...], dict[str, dict[str, bool]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    for row in final:
        cell = (
            row["budget_protocol"],
            row["model"],
            row["seed"],
        )
        by_cell[cell][row["method"]][str(row["question_id"])] = bool(
            row["is_correct"]
        )
    comparisons: dict[tuple[str, str, str], dict[int, list[tuple[bool, bool]]]] = defaultdict(dict)
    for (protocol, model, seed), methods in by_cell.items():
        baseline = methods.get("base")
        if not baseline:
            continue
        for method, answers in methods.items():
            if method == "base":
                continue
            shared = sorted(set(baseline) & set(answers))
            comparisons[(protocol, model, method)][int(seed)] = [
                (baseline[key], answers[key]) for key in shared
            ]
    output = []
    for (protocol, model, method), by_seed in sorted(comparisons.items()):
        pooled = [pair for pairs in by_seed.values() for pair in pairs]
        test = mcnemar_test(pooled)
        paired = paired_bootstrap(pooled)
        stratified = stratified_seed_item_bootstrap(by_seed)
        output.append(
            {
                "budget_protocol": protocol,
                "model": model,
                "baseline_method": "base",
                "method": method,
                **test,
                "paired_bootstrap_ci95_low": paired["ci95_low"],
                "paired_bootstrap_ci95_high": paired["ci95_high"],
                "stratified_bootstrap_ci95_low": stratified["ci95_low"],
                "stratified_bootstrap_ci95_high": stratified["ci95_high"],
            }
        )
    adjusted = holm_correction(item["p_value"] for item in output)
    for item, value in zip(output, adjusted):
        item["holm_adjusted_p_value"] = value
    return output


def _write_csv(path: Path, rows: list[Mapping[str, Any]], fields: Iterable[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields or (rows[0].keys() if rows else []))
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def build_paper_tables(index_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    long_rows, run_rows = load_results_long(index_path)
    final = _final_rows(long_rows)
    item_metrics = _run_item_metrics(long_rows)
    for row in run_rows:
        row.update(item_metrics.get(row["study_id"], {}))

    group_values: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for row in run_rows:
        if row.get("final_accuracy") is not None:
            group_values[
                (
                    row["budget_protocol"],
                    row["model"],
                    row["method"],
                )
            ].append(float(row["final_accuracy"]))
    main_results = [
        {
            "budget_protocol": protocol,
            "model": model,
            "method": method,
            **summarize_values(values),
        }
        for (protocol, model, method), values in sorted(group_values.items())
    ]
    base_means = {
        (row["budget_protocol"], row["model"]): row["mean"]
        for row in main_results
        if row["method"] == "base"
    }
    run_groups: dict[
        tuple[str, str, str],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for run in run_rows:
        run_groups[
            (
                run["budget_protocol"],
                run["model"],
                run["method"],
            )
        ].append(run)
    for row in main_results:
        base_mean = base_means.get(
            (row["budget_protocol"], row["model"])
        )
        row["base_mean_accuracy"] = base_mean
        row["mean_delta_vs_base"] = (
            float(row["mean"]) - float(base_mean)
            if row["mean"] is not None and base_mean is not None
            else None
        )
        runs = run_groups[
            (
                row["budget_protocol"],
                row["model"],
                row["method"],
            )
        ]
        for metric in (
            "macro_accuracy",
            "worst_subcategory_accuracy",
            "learning_curve_auc",
            "forgetting_rate",
            "retention_accuracy_delta",
            "retention_forgetting_rate",
        ):
            values = [
                float(run[metric])
                for run in runs
                if run.get(metric) is not None
            ]
            row[f"{metric}_mean"] = (
                sum(values) / len(values) if values else None
            )
    ablation_methods = {
        "full",
        "full_no_hard_pool",
        "full_no_error_targeting",
        "full_no_observed_difficulty_sampling",
        "full_no_difficulty_module",
        "full_no_coverage_priority",
        "full_no_uncertainty_priority",
        "full_no_global_difficulty",
        "full_no_retention_priority",
    }
    ablations = [
        row for row in main_results if row["method"] in ablation_methods
    ]
    category = _accuracy_rows(
        final,
        ("budget_protocol", "model", "method", "category"),
    )
    difficulty = _accuracy_rows(
        final,
        ("budget_protocol", "model", "method", "difficulty"),
    )
    efficiency = [
        {
            key: row.get(key)
            for key in (
                "study_id",
                "budget_protocol",
                "model",
                "method",
                "seed",
                "training_sample_count",
                "training_token_count",
                "gpu_hours",
                "total_tokens",
                "total_api_calls",
                "estimated_cost",
                "accepted_samples_per_1k_generated_tokens",
                "accuracy_gain_per_1k_training_tokens",
                "accuracy_gain_per_gpu_hour",
            )
        }
        for row in run_rows
    ]
    significance = _significance(long_rows)
    tables = {
        "results_long.csv": (long_rows, LONG_FIELDS),
        "run_summary.csv": (run_rows, None),
        "main_results.csv": (main_results, None),
        "ablation_results.csv": (ablations, None),
        "category_results.csv": (category, None),
        "difficulty_results.csv": (difficulty, None),
        "efficiency_results.csv": (efficiency, None),
        "significance_tests.csv": (significance, None),
    }
    for name, (rows, fields) in tables.items():
        _write_csv(output / name, rows, fields)
    return {
        "output_dir": str(output),
        "raw_item_rows": len(long_rows),
        "run_count": len(run_rows),
        "table_files": list(tables),
    }
