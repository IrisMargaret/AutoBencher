"""Small dependency-free statistics helpers for study aggregation."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping


_T_975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    21: 2.080, 22: 2.074, 23: 2.069, 24: 2.064, 25: 2.060,
    26: 2.056, 27: 2.052, 28: 2.048, 29: 2.045, 30: 2.042,
}


def _student_t_critical_95(sample_count: int) -> float:
    """Two-sided 95% t critical value without a SciPy dependency."""
    degrees = int(sample_count) - 1
    if degrees in _T_975:
        return _T_975[degrees]
    if degrees <= 0:
        raise ValueError("Student-t interval requires at least two samples")
    # Cornish-Fisher expansion around the normal 97.5th percentile. At
    # df>30 its error is negligible for the reported table precision.
    z = 1.959963984540054
    inverse = 1.0 / degrees
    return (
        z
        + (z**3 + z) * inverse / 4
        + (5 * z**5 + 16 * z**3 + 3 * z) * inverse**2 / 96
        + (3 * z**7 + 19 * z**5 + 17 * z**3 - 15 * z)
        * inverse**3
        / 384
    )


def summarize_values(values: Iterable[float]) -> dict[str, float | int | None]:
    samples = [float(value) for value in values]
    if not samples:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "ci95_low": None,
            "ci95_high": None,
            "ci_method": "undefined_empty_sample",
        }
    mean = statistics.fmean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    critical = _student_t_critical_95(len(samples)) if len(samples) > 1 else None
    margin = critical * std / math.sqrt(len(samples)) if critical else 0.0
    return {
        "n": len(samples),
        "mean": mean,
        "std": std,
        "min": min(samples),
        "max": max(samples),
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
        "ci_method": "student_t_across_seeds" if len(samples) > 1 else "single_seed_no_uncertainty",
        "critical_value": critical,
    }


def aggregate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    metric: str = "final_accuracy",
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[float]] = defaultdict(list)
    for record in records:
        value = record.get(metric)
        if value is None:
            continue
        key = (
            str(record.get("evaluation_set_id", "unknown")),
            str(record.get("evaluation_set_version", "unknown")),
            str(record.get("evaluation_set_sha256", "unknown")),
            str(record.get("budget_protocol", "question_matched")),
            str(record["method"]),
            str(record.get("variant", record["method"])),
            str(record["model"]),
            int(record["budget"]),
        )
        groups[key].append(float(value))
    output = []
    for (
        evaluation_set_id,
        evaluation_set_version,
        evaluation_set_sha256,
        budget_protocol,
        method,
        variant,
        model,
        budget,
    ), values in sorted(groups.items()):
        output.append(
            {
                "evaluation_set_id": evaluation_set_id,
                "evaluation_set_version": evaluation_set_version,
                "evaluation_set_sha256": evaluation_set_sha256,
                "budget_protocol": budget_protocol,
                "method": method,
                "variant": variant,
                "model": model,
                "budget": budget,
                "metric": metric,
                **summarize_values(values),
            }
        )
    return output


def mcnemar_test(pairs: Iterable[tuple[bool, bool]]) -> dict[str, Any]:
    pairs = list(pairs)
    baseline_only = sum(left and not right for left, right in pairs)
    method_only = sum(right and not left for left, right in pairs)
    discordant = baseline_only + method_only
    if discordant:
        tail = sum(
            math.comb(discordant, value)
            for value in range(0, min(baseline_only, method_only) + 1)
        ) / (2 ** discordant)
        p_value = min(1.0, 2 * tail)
    else:
        p_value = 1.0
    return {
        "n": len(pairs),
        "baseline_only_correct": baseline_only,
        "method_only_correct": method_only,
        "discordant": discordant,
        "p_value": p_value,
        "risk_difference": (
            sum(right for _, right in pairs) / len(pairs)
            - sum(left for left, _ in pairs) / len(pairs)
            if pairs
            else 0.0
        ),
        "matched_odds_ratio": (
            (method_only + 0.5) / (baseline_only + 0.5)
        ),
    }


def paired_bootstrap(
    pairs: Iterable[tuple[bool, bool]],
    *,
    iterations: int = 2000,
    seed: int = 2026,
) -> dict[str, float | int]:
    pairs = list(pairs)
    if not pairs:
        return {"iterations": iterations, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    effects = []
    for _ in range(iterations):
        sample = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        effects.append(
            sum(int(right) - int(left) for left, right in sample)
            / len(sample)
        )
    effects.sort()
    lower = effects[int(0.025 * (iterations - 1))]
    upper = effects[int(0.975 * (iterations - 1))]
    return {
        "iterations": iterations,
        "ci95_low": lower,
        "ci95_high": upper,
    }


def stratified_seed_item_bootstrap(
    pairs_by_seed: Mapping[int, list[tuple[bool, bool]]],
    *,
    iterations: int = 2000,
    seed: int = 2026,
) -> dict[str, float | int]:
    seeds = sorted(pairs_by_seed)
    if not seeds:
        return {"iterations": iterations, "ci95_low": 0.0, "ci95_high": 0.0}
    rng = random.Random(seed)
    effects = []
    for _ in range(iterations):
        sampled_seeds = [seeds[rng.randrange(len(seeds))] for _ in seeds]
        sampled_pairs = []
        for sampled_seed in sampled_seeds:
            items = pairs_by_seed[sampled_seed]
            sampled_pairs.extend(
                items[rng.randrange(len(items))] for _ in items
            )
        effects.append(
            sum(int(right) - int(left) for left, right in sampled_pairs)
            / len(sampled_pairs)
        )
    effects.sort()
    return {
        "iterations": iterations,
        "ci95_low": effects[int(0.025 * (iterations - 1))],
        "ci95_high": effects[int(0.975 * (iterations - 1))],
    }


def holm_correction(p_values: Iterable[float]) -> list[float]:
    values = [float(value) for value in p_values]
    ordered = sorted(enumerate(values), key=lambda item: item[1])
    adjusted = [1.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, (index, value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * value)
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted
