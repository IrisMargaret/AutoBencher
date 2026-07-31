"""Rebuild deterministic, paper-ready tables from raw study artifacts."""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from autobencher.result_schema import ExperimentRecord, validate_registry
from autobencher.study_runner import (
    StudyConfigurationError,
    validate_experiment_completion,
)
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
    "budget",
    "evaluation_set_id",
    "evaluation_set_version",
    "evaluation_set_sha256",
    "checkpoint_sha256",
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


def load_results_long(
    index_path: str | Path,
    *,
    allow_partial_development_results: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index_path = Path(index_path)
    registry = _read_json(index_path)
    validate_registry(registry)
    rows = []
    run_rows = []
    records = [
        ExperimentRecord.from_dict(payload)
        for payload in registry["experiments"]
    ]
    if not allow_partial_development_results:
        incomplete = [record.study_id for record in records if record.status != "completed"]
        if incomplete:
            raise StudyConfigurationError(
                "Paper aggregation requires every preregistered experiment "
                f"to be completed: {incomplete}"
            )
    seed_sets: dict[tuple[Any, ...], set[int]] = defaultdict(set)
    for record in records:
        if record.status == "completed":
            seed_sets[
                (
                    record.evaluation_set_id,
                    record.evaluation_set_version,
                    record.evaluation_set_sha256,
                    record.budget_protocol,
                    record.model,
                    record.budget,
                    record.method,
                    record.variant,
                )
            ].add(record.seed)
    expected_seed_sets = {frozenset(value) for value in seed_sets.values()}
    if not allow_partial_development_results and len(expected_seed_sets) > 1:
        raise StudyConfigurationError("Paper cells have inconsistent seed sets")
    for record in records:
        if record.status != "completed":
            continue
        try:
            completion = validate_experiment_completion(record)
        except StudyConfigurationError:
            if allow_partial_development_results:
                continue
            raise
        run_dir = Path(record.run_dir).resolve()
        summary = {}
        ledger = {}
        if run_dir:
            summary_path = Path(record.summary_path).resolve()
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
                            "budget": record.budget,
                            "evaluation_set_id": completion["evaluation_set_id"],
                            "evaluation_set_version": completion["evaluation_set_version"],
                            "evaluation_set_sha256": completion["evaluation_set_sha256"],
                            "checkpoint_sha256": completion["checkpoint_sha256"],
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
                "evaluation_set_id": completion["evaluation_set_id"],
                "evaluation_set_version": completion["evaluation_set_version"],
                "evaluation_set_sha256": completion["evaluation_set_sha256"],
                "checkpoint_sha256": completion["checkpoint_sha256"],
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
                "selected_pool_count": training.get("selected_pool_count"),
                "train_count": training.get("train_count"),
                "validation_count": training.get("validation_count"),
                "internal_test_count": training.get("internal_test_count"),
                "actual_trained_count": training.get(
                    "actual_trained_count"
                ),
                "train_correct_count": training.get("train_correct_count"),
                "train_incorrect_count": training.get(
                    "train_incorrect_count"
                ),
                "optimizer_steps": training.get("optimizer_steps"),
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


def _combine_seed_p_values(values: Iterable[float]) -> float:
    probabilities = [max(float(value), 1.0e-300) for value in values]
    if not probabilities:
        return 1.0
    statistic = -2.0 * sum(math.log(value) for value in probabilities)
    half = statistic / 2.0
    # Chi-square survival function for 2*k degrees of freedom.
    return min(
        1.0,
        math.exp(-half)
        * sum(half ** index / math.factorial(index) for index in range(len(probabilities))),
    )


def _significance(
    rows: list[dict[str, Any]],
    comparison_pairs: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    final = _final_rows(rows)
    by_cell: dict[tuple[Any, ...], dict[str, dict[str, bool]]] = defaultdict(
        lambda: defaultdict(dict)
    )
    variants_by_cell: dict[tuple[Any, ...], dict[str, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for row in final:
        cell = (
            row["evaluation_set_id"],
            row["evaluation_set_version"],
            row["evaluation_set_sha256"],
            row["budget_protocol"],
            row["model"],
            row["budget"],
            row["seed"],
        )
        by_cell[cell][row["method"]][str(row["question_id"])] = bool(
            row["is_correct"]
        )
        variants_by_cell[cell][row["method"]].add(str(row["variant"]))
    registered = []
    for item in comparison_pairs:
        left = str(item.get("left", item.get("baseline", "")))
        right = str(item.get("right", item.get("method", "")))
        if not left or not right or left == right:
            raise StudyConfigurationError(f"Invalid comparison pair: {item}")
        registered.append((left, right))
    if not registered:
        raise StudyConfigurationError("Paper mode requires preregistered comparison_pairs")
    comparisons: dict[tuple[Any, ...], dict[int, list[tuple[bool, bool]]]] = defaultdict(dict)
    for (
        evaluation_id,
        evaluation_version,
        evaluation_sha,
        protocol,
        model,
        budget,
        seed,
    ), methods in by_cell.items():
        for left_method, right_method in registered:
            left = methods.get(left_method)
            right = methods.get(right_method)
            if left is None or right is None:
                raise StudyConfigurationError(
                    f"Missing preregistered comparison {left_method} vs {right_method} "
                    f"in cell {(evaluation_id, protocol, model, budget, seed)}"
                )
            if set(left) != set(right):
                raise StudyConfigurationError(
                    f"Paired item IDs differ for {left_method} vs {right_method}; "
                    "formal aggregation never takes a silent intersection"
                )
            left_variants = variants_by_cell[
                (
                    evaluation_id,
                    evaluation_version,
                    evaluation_sha,
                    protocol,
                    model,
                    budget,
                    seed,
                )
            ][left_method]
            right_variants = variants_by_cell[
                (
                    evaluation_id,
                    evaluation_version,
                    evaluation_sha,
                    protocol,
                    model,
                    budget,
                    seed,
                )
            ][right_method]
            if len(left_variants) != 1 or len(right_variants) != 1:
                raise StudyConfigurationError(
                    "A formal comparison cell must contain one variant per method"
                )
            identifiers = sorted(left)
            comparisons[(
                evaluation_id,
                evaluation_version,
                evaluation_sha,
                protocol,
                model,
                budget,
                left_method,
                next(iter(left_variants)),
                right_method,
                next(iter(right_variants)),
            )][int(seed)] = [
                (left[key], right[key]) for key in identifiers
            ]
    output = []
    for key, by_seed in sorted(comparisons.items()):
        (
            evaluation_id,
            evaluation_version,
            evaluation_sha,
            protocol,
            model,
            budget,
            left_method,
            left_variant,
            right_method,
            right_variant,
        ) = key
        pooled = [pair for pairs in by_seed.values() for pair in pairs]
        descriptive = mcnemar_test(pooled)
        seed_tests = {
            seed: mcnemar_test(pairs) for seed, pairs in sorted(by_seed.items())
        }
        primary_p = _combine_seed_p_values(
            result["p_value"] for result in seed_tests.values()
        )
        primary_effect = sum(
            result["risk_difference"] for result in seed_tests.values()
        ) / len(seed_tests)
        paired = paired_bootstrap(pooled)
        stratified = stratified_seed_item_bootstrap(by_seed)
        output.append(
            {
                "budget_protocol": protocol,
                "model": model,
                "budget": budget,
                "evaluation_set_id": evaluation_id,
                "evaluation_set_version": evaluation_version,
                "evaluation_set_sha256": evaluation_sha,
                "baseline_method": left_method,
                "baseline_variant": left_variant,
                "method": right_method,
                "variant": right_variant,
                "test": "per_seed_mcnemar_fisher_combination",
                "seed_count": len(seed_tests),
                "p_value": primary_p,
                "effect_size_mean_seed_risk_difference": primary_effect,
                "pooled_risk_difference_descriptive_only": descriptive[
                    "risk_difference"
                ],
                "pooled_mcnemar_p_value_descriptive_only": descriptive["p_value"],
                "per_seed_tests_json": json.dumps(seed_tests, sort_keys=True),
                "pooled_item_bootstrap_ci95_low_descriptive_only": paired[
                    "ci95_low"
                ],
                "pooled_item_bootstrap_ci95_high_descriptive_only": paired[
                    "ci95_high"
                ],
                "primary_cluster_bootstrap_ci95_low": stratified["ci95_low"],
                "primary_cluster_bootstrap_ci95_high": stratified["ci95_high"],
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


def build_paper_tables(
    index_path: str | Path,
    output_dir: str | Path,
    *,
    allow_partial_development_results: bool = False,
) -> dict[str, Any]:
    output = Path(output_dir)
    registry = _read_json(Path(index_path))
    long_rows, run_rows = load_results_long(
        index_path,
        allow_partial_development_results=allow_partial_development_results,
    )
    final = _final_rows(long_rows)
    item_metrics = _run_item_metrics(long_rows)
    for row in run_rows:
        row.update(item_metrics.get(row["study_id"], {}))

    group_values: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for row in run_rows:
        if row.get("final_accuracy") is not None:
            group_values[
                (
                    row["evaluation_set_id"],
                    row["evaluation_set_version"],
                    row["evaluation_set_sha256"],
                    row["budget_protocol"],
                    row["model"],
                    row["budget"],
                    row["method"],
                    row["variant"],
                )
            ].append(float(row["final_accuracy"]))
    main_results = [
        {
            "evaluation_set_id": evaluation_id,
            "evaluation_set_version": evaluation_version,
            "evaluation_set_sha256": evaluation_sha,
            "budget_protocol": protocol,
            "model": model,
            "budget": budget,
            "method": method,
            "variant": variant,
            **summarize_values(values),
        }
        for (
            evaluation_id,
            evaluation_version,
            evaluation_sha,
            protocol,
            model,
            budget,
            method,
            variant,
        ), values in sorted(group_values.items())
    ]
    base_means = {
        (
            row["evaluation_set_id"],
            row["evaluation_set_sha256"],
            row["evaluation_set_version"],
            row["budget_protocol"],
            row["model"],
            row["budget"],
        ): row["mean"]
        for row in main_results
        if row["method"] == "base"
    }
    run_groups: dict[
        tuple[Any, ...],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for run in run_rows:
        run_groups[
            (
                run["budget_protocol"],
                run["model"],
                run["budget"],
                run["method"],
                run["variant"],
                run["evaluation_set_id"],
                run["evaluation_set_sha256"],
                run["evaluation_set_version"],
            )
        ].append(run)
    for row in main_results:
        base_mean = base_means.get(
            (
                row["evaluation_set_id"],
                row["evaluation_set_sha256"],
                row["evaluation_set_version"],
                row["budget_protocol"],
                row["model"],
                row["budget"],
            )
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
                row["budget"],
                row["method"],
                row["variant"],
                row["evaluation_set_id"],
                row["evaluation_set_sha256"],
                row["evaluation_set_version"],
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
    category_by_seed = _accuracy_rows(
        final,
        (
            "evaluation_set_id",
            "evaluation_set_version",
            "evaluation_set_sha256",
            "budget_protocol",
            "model",
            "budget",
            "method",
            "variant",
            "seed",
            "category",
        ),
    )
    difficulty_by_seed = _accuracy_rows(
        final,
        (
            "evaluation_set_id",
            "evaluation_set_version",
            "evaluation_set_sha256",
            "budget_protocol",
            "model",
            "budget",
            "method",
            "variant",
            "seed",
            "difficulty",
        ),
    )
    def across_seed_table(
        per_seed: list[dict[str, Any]], dimension: str
    ) -> list[dict[str, Any]]:
        identity_fields = (
            "evaluation_set_id",
            "evaluation_set_version",
            "evaluation_set_sha256",
            "budget_protocol",
            "model",
            "budget",
            "method",
            "variant",
            dimension,
        )
        values: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
        for item in per_seed:
            values[tuple(item[field] for field in identity_fields)].append(item)
        return [
            {
                **dict(zip(identity_fields, key)),
                "seed_count": len(items),
                "seeds_json": json.dumps(
                    sorted(int(item["seed"]) for item in items)
                ),
                "per_seed_question_count": items[0]["question_count"],
                **summarize_values(item["accuracy"] for item in items),
            }
            for key, items in sorted(values.items(), key=lambda item: str(item[0]))
        ]
    category = across_seed_table(category_by_seed, "category")
    difficulty = across_seed_table(difficulty_by_seed, "difficulty")
    efficiency = [
        {
            key: row.get(key)
            for key in (
                "study_id",
                "budget_protocol",
                "budget",
                "evaluation_set_id",
                "evaluation_set_version",
                "evaluation_set_sha256",
                "checkpoint_sha256",
                "model",
                "method",
                "variant",
                "seed",
                "training_sample_count",
                "selected_pool_count",
                "train_count",
                "validation_count",
                "internal_test_count",
                "actual_trained_count",
                "train_correct_count",
                "train_incorrect_count",
                "training_token_count",
                "optimizer_steps",
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
    comparison_pairs = registry.get("comparison_pairs", [])
    significance = (
        _significance(long_rows, comparison_pairs)
        if comparison_pairs
        else (
            []
            if allow_partial_development_results
            else _significance(long_rows, comparison_pairs)
        )
    )
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
