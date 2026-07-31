"""Small dependency-free statistics helpers for study aggregation."""

from __future__ import annotations

import math
import statistics
from collections import defaultdict
from typing import Any, Iterable, Mapping


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
        }
    mean = statistics.fmean(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    margin = 1.96 * std / math.sqrt(len(samples)) if len(samples) > 1 else 0.0
    return {
        "n": len(samples),
        "mean": mean,
        "std": std,
        "min": min(samples),
        "max": max(samples),
        "ci95_low": mean - margin,
        "ci95_high": mean + margin,
    }


def aggregate_records(
    records: Iterable[Mapping[str, Any]],
    *,
    metric: str = "final_accuracy",
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[float]] = defaultdict(list)
    for record in records:
        value = record.get(metric)
        if value is None:
            continue
        key = (
            str(record["method"]),
            str(record["model"]),
            int(record["budget"]),
        )
        groups[key].append(float(value))
    output = []
    for (method, model, budget), values in sorted(groups.items()):
        output.append(
            {
                "method": method,
                "model": model,
                "budget": budget,
                "metric": metric,
                **summarize_values(values),
            }
        )
    return output
