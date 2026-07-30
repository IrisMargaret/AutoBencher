"""Human-review exports and error-attribution agreement metrics."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .experiment import atomic_json


def _evidence_items(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    value = row.get("evidence_json", "")
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    try:
        parsed = json.loads(str(value or "[]"))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return (
        [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, list)
        else []
    )


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
        "attribution_confidence",
        "needs_review",
        "verification_tier",
        "first_error_step",
        "evidence_json",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in selected:
            row = {
                field: record.get(field, "")
                for field in fields
                if field != "evidence_json"
            }
            row["evidence_json"] = json.dumps(
                record.get("evidence", []),
                ensure_ascii=False,
            )
            writer.writerow(row)
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
    reviewed_pairs = [
        row
        for row in rows
        if row.get("primary_error_tag") and row.get("human_label_1")
    ]
    non_abstained = [
        row
        for row in reviewed_pairs
        if row.get("primary_error_tag") != "unknown_error"
    ]
    high_confidence = []
    calibration_pairs = []
    accuracy_by_tier = defaultdict(list)
    for row in reviewed_pairs:
        accuracy_by_tier[
            row.get("verification_tier", "") or "unspecified"
        ].append(
            float(
                row.get("primary_error_tag")
                == row.get("human_label_1")
            )
        )
        try:
            confidence = float(row.get("attribution_confidence", ""))
        except (TypeError, ValueError):
            continue
        confidence = max(0.0, min(1.0, confidence))
        correct = float(
            row.get("primary_error_tag") == row.get("human_label_1")
        )
        calibration_pairs.append((confidence, correct))
        if confidence >= 0.8:
            high_confidence.append((confidence, correct))
    result.update(
        {
            # Ragas-style composable quality signals: correctness, coverage,
            # evidence completeness, and calibration are reported separately.
            "attribution_accuracy": (
                sum(
                    row.get("primary_error_tag")
                    == row.get("human_label_1")
                    for row in reviewed_pairs
                )
                / len(reviewed_pairs)
                if reviewed_pairs
                else None
            ),
            "selective_coverage": (
                len(non_abstained) / len(reviewed_pairs)
                if reviewed_pairs
                else None
            ),
            "selective_accuracy": (
                sum(
                    row.get("primary_error_tag")
                    == row.get("human_label_1")
                    for row in non_abstained
                )
                / len(non_abstained)
                if non_abstained
                else None
            ),
            "macro_f1": (
                sum(item["f1"] for item in per_label.values())
                / len(per_label)
                if per_label
                else None
            ),
            "evidence_coverage": (
                sum(bool(_evidence_items(row)) for row in reviewed_pairs)
                / len(reviewed_pairs)
                if reviewed_pairs
                else None
            ),
            "verified_evidence_coverage": (
                sum(
                    any(
                        bool(item.get("check_name"))
                        for item in _evidence_items(row)
                    )
                    for row in reviewed_pairs
                )
                / len(reviewed_pairs)
                if reviewed_pairs
                else None
            ),
            "confidence_brier_score": (
                sum(
                    (confidence - correct) ** 2
                    for confidence, correct in calibration_pairs
                )
                / len(calibration_pairs)
                if calibration_pairs
                else None
            ),
            "high_confidence_accuracy": (
                sum(correct for _, correct in high_confidence)
                / len(high_confidence)
                if high_confidence
                else None
            ),
            "high_confidence_count": len(high_confidence),
            "accuracy_by_verification_tier": {
                tier: sum(values) / len(values)
                for tier, values in sorted(accuracy_by_tier.items())
            },
        }
    )
    if json_output:
        atomic_json(result, json_output)
    return result
