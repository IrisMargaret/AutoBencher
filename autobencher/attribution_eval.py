"""Human-review exports and error-attribution agreement metrics."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .experiment import atomic_json


def export_review_sample(
    records: Iterable[Mapping[str, Any]],
    csv_path: str | Path,
    sample_size: int = 100,
    seed: int = 42,
) -> int:
    errors = [dict(record) for record in records if not record.get("is_correct")]
    random.Random(seed).shuffle(errors)
    selected = errors[:sample_size]
    path = Path(csv_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "question_id",
        "question",
        "gold_answer",
        "test_taker_response",
        "primary_error_tag",
        "human_label_1",
        "human_label_2",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in selected:
            writer.writerow(
                {
                    field: record.get(field, "")
                    for field in fields
                }
            )
    return len(selected)


def _cohen_kappa(left: list[str], right: list[str]) -> float | None:
    pairs = [
        (a, b)
        for a, b in zip(left, right)
        if a and b
    ]
    if not pairs:
        return None
    labels = sorted({value for pair in pairs for value in pair})
    observed = sum(a == b for a, b in pairs) / len(pairs)
    left_counts = Counter(a for a, _ in pairs)
    right_counts = Counter(b for _, b in pairs)
    expected = sum(
        left_counts[label] / len(pairs) * right_counts[label] / len(pairs)
        for label in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def evaluate_review_csv(
    csv_path: str | Path,
    json_output: str | Path | None = None,
) -> dict[str, Any]:
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    labels = sorted(
        {
            value
            for row in rows
            for value in (
                row.get("primary_error_tag", ""),
                row.get("human_label_1", ""),
            )
            if value
        }
    )
    confusion = defaultdict(Counter)
    for row in rows:
        predicted = row.get("primary_error_tag", "")
        actual = row.get("human_label_1", "")
        if predicted and actual:
            confusion[actual][predicted] += 1
    per_label = {}
    for label in labels:
        true_positive = confusion[label][label]
        false_positive = sum(
            confusion[actual][label]
            for actual in labels
            if actual != label
        )
        false_negative = sum(
            confusion[label][predicted]
            for predicted in labels
            if predicted != label
        )
        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_label[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    result = {
        "reviewed_count": sum(
            bool(row.get("primary_error_tag") and row.get("human_label_1"))
            for row in rows
        ),
        "exported_row_count": len(rows),
        "labels": labels,
        "per_label": per_label,
        "confusion_matrix": {
            actual: {
                predicted: confusion[actual][predicted]
                for predicted in labels
            }
            for actual in labels
        },
        "cohen_kappa": _cohen_kappa(
            [row.get("human_label_1", "") for row in rows],
            [row.get("human_label_2", "") for row in rows],
        ),
    }
    if json_output:
        atomic_json(result, json_output)
    return result
