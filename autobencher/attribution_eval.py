"""Human-review exports and error-attribution agreement metrics."""

from __future__ import annotations

import csv
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .experiment import atomic_json, utc_now
from .fingerprints import file_sha256


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
        "adjudicated_label",
        "adjudication_notes",
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


def export_blinded_review_packets(
    records: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    sample_size: int = 400,
    seed: int = 42,
) -> dict[str, Any]:
    """Export two independent blind packets and a sealed system prediction file."""

    errors = [dict(record) for record in records if not record.get("is_correct")]
    rng = random.Random(seed)
    rng.shuffle(errors)
    selected = errors[:sample_size]
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    public_fields = [
        "question_id",
        "question",
        "gold_answer",
        "test_taker_response",
        "verification_tier",
        "first_error_step",
        "evidence_json",
        "human_label",
        "annotator_notes",
    ]
    system_fields = [
        "question_id",
        "primary_error_tag",
        "attribution_confidence",
        "needs_review",
        "verification_tier",
        "evidence_json",
    ]
    system_path = root / "system_predictions.sealed.csv"
    with system_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=system_fields)
        writer.writeheader()
        for record in selected:
            writer.writerow(
                {
                    "question_id": _record_id(record),
                    "primary_error_tag": record.get("primary_error_tag", ""),
                    "attribution_confidence": record.get(
                        "attribution_confidence", ""
                    ),
                    "needs_review": record.get("needs_review", ""),
                    "verification_tier": record.get("verification_tier", ""),
                    "evidence_json": json.dumps(
                        record.get("evidence", []), ensure_ascii=False
                    ),
                }
            )
    packet_paths = []
    for annotator_number in (1, 2):
        packet_records = list(selected)
        random.Random(seed + annotator_number * 104729).shuffle(packet_records)
        packet_path = root / f"annotator_{annotator_number}.csv"
        packet_paths.append(packet_path)
        with packet_path.open(
            "w", encoding="utf-8-sig", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=public_fields)
            writer.writeheader()
            for record in packet_records:
                writer.writerow(
                    {
                        "question_id": _record_id(record),
                        "question": record.get("question", ""),
                        "gold_answer": record.get("gold_answer", ""),
                        "test_taker_response": record.get(
                            "test_taker_response",
                            record.get("predicted_answer", ""),
                        ),
                        "verification_tier": record.get(
                            "verification_tier", ""
                        ),
                        "first_error_step": record.get(
                            "first_error_step", ""
                        ),
                        "evidence_json": json.dumps(
                            record.get("evidence", []), ensure_ascii=False
                        ),
                        "human_label": "",
                        "annotator_notes": "",
                    }
                )
    manifest = {
        "schema_version": "1.0",
        "created_at": utc_now(),
        "protocol": "two_independent_blind_annotators_plus_adjudication",
        "requested_sample_size": sample_size,
        "sample_size": len(selected),
        "seed": seed,
        "blinding": {
            "annotators_can_see_system_label": False,
            "annotators_can_see_each_other_label": False,
            "packet_order_is_independent": True,
        },
        "files": {
            "system_predictions": {
                "path": system_path.name,
                "sha256": file_sha256(system_path),
            },
            "annotator_1": {
                "path": packet_paths[0].name,
                "sha256": file_sha256(packet_paths[0]),
            },
            "annotator_2": {
                "path": packet_paths[1].name,
                "sha256": file_sha256(packet_paths[1]),
            },
        },
    }
    atomic_json(manifest, root / "review_manifest.json")
    return manifest


def _record_id(record: Mapping[str, Any]) -> str:
    value = record.get("question_id", record.get("id", ""))
    if not str(value).strip():
        raise ValueError("Every review record must have a question_id.")
    return str(value)


def _csv_by_id(path: str | Path) -> dict[str, dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = {}
    for row in rows:
        question_id = str(row.get("question_id", "")).strip()
        if not question_id or question_id in result:
            raise ValueError(f"Missing or duplicate question_id in {path}.")
        result[question_id] = row
    return result


def merge_blinded_reviews(
    system_predictions: str | Path,
    annotator_1: str | Path,
    annotator_2: str | Path,
    output_csv: str | Path,
) -> dict[str, Any]:
    """Merge blind reviews without silently resolving disagreements."""

    system = _csv_by_id(system_predictions)
    first = _csv_by_id(annotator_1)
    second = _csv_by_id(annotator_2)
    if set(system) != set(first) or set(system) != set(second):
        raise ValueError("System and annotator packets must contain identical IDs.")
    fields = [
        "question_id",
        "primary_error_tag",
        "human_label_1",
        "human_label_2",
        "adjudicated_label",
        "adjudication_notes",
        "attribution_confidence",
        "needs_review",
        "verification_tier",
        "evidence_json",
    ]
    path = Path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    agreements = 0
    disagreements = 0
    incomplete = 0
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for question_id in sorted(system):
            label_1 = first[question_id].get("human_label", "").strip()
            label_2 = second[question_id].get("human_label", "").strip()
            if not label_1 or not label_2:
                incomplete += 1
            elif label_1 == label_2:
                agreements += 1
            else:
                disagreements += 1
            source = system[question_id]
            writer.writerow(
                {
                    "question_id": question_id,
                    "primary_error_tag": source.get("primary_error_tag", ""),
                    "human_label_1": label_1,
                    "human_label_2": label_2,
                    # Agreement is already adjudicated; disagreements remain
                    # blank for a third reviewer or documented rule.
                    "adjudicated_label": (
                        label_1 if label_1 and label_1 == label_2 else ""
                    ),
                    "adjudication_notes": (
                        "automatic_consensus"
                        if label_1 and label_1 == label_2
                        else ""
                    ),
                    "attribution_confidence": source.get(
                        "attribution_confidence", ""
                    ),
                    "needs_review": source.get("needs_review", ""),
                    "verification_tier": source.get(
                        "verification_tier", ""
                    ),
                    "evidence_json": source.get("evidence_json", "[]"),
                }
            )
    return {
        "row_count": len(system),
        "agreement_count": agreements,
        "disagreement_count": disagreements,
        "incomplete_count": incomplete,
        "requires_adjudication": disagreements + incomplete,
        "output": str(path),
    }


def _gold_label(row: Mapping[str, Any]) -> str:
    adjudicated = str(row.get("adjudicated_label", "")).strip()
    if adjudicated:
        return adjudicated
    first = str(row.get("human_label_1", "")).strip()
    second = str(row.get("human_label_2", "")).strip()
    return first if first and first == second else ""


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
                _gold_label(row),
            )
            if value
        }
    )
    confusion = defaultdict(Counter)
    for row in rows:
        predicted = row.get("primary_error_tag", "")
        actual = _gold_label(row)
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
            bool(row.get("primary_error_tag") and _gold_label(row))
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
        if row.get("primary_error_tag") and _gold_label(row)
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
                == _gold_label(row)
            )
        )
        try:
            confidence = float(row.get("attribution_confidence", ""))
        except (TypeError, ValueError):
            continue
        confidence = max(0.0, min(1.0, confidence))
        correct = float(
            row.get("primary_error_tag") == _gold_label(row)
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
                    == _gold_label(row)
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
                    == _gold_label(row)
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
            "unknown_error_ratio": (
                sum(
                    row.get("primary_error_tag") == "unknown_error"
                    for row in reviewed_pairs
                )
                / len(reviewed_pairs)
                if reviewed_pairs
                else None
            ),
            "unknown_error_policy": (
                "unknown_error is a deliberate abstention when deterministic "
                "evidence cannot isolate one mechanism; report its ratio "
                "instead of forcing a low-confidence class."
            ),
            "unresolved_adjudication_count": sum(
                bool(row.get("human_label_1"))
                and bool(row.get("human_label_2"))
                and not _gold_label(row)
                for row in rows
            ),
            "accuracy_by_verification_tier": {
                tier: sum(values) / len(values)
                for tier, values in sorted(accuracy_by_tier.items())
            },
        }
    )
    kappa = result["cohen_kappa"]
    high_accuracy = result["high_confidence_accuracy"]
    result["acceptance"] = {
        "minimum_reviewed_count": 300,
        "minimum_cohen_kappa": 0.70,
        "minimum_high_confidence_accuracy": 0.80,
        "reviewed_count_pass": result["reviewed_count"] >= 300,
        "cohen_kappa_pass": kappa is not None and kappa >= 0.70,
        "high_confidence_accuracy_pass": (
            high_accuracy is not None and high_accuracy >= 0.80
        ),
    }
    result["acceptance"]["paper_core_signal_ready"] = all(
        (
            result["acceptance"]["reviewed_count_pass"],
            result["acceptance"]["cohen_kappa_pass"],
            result["acceptance"]["high_confidence_accuracy_pass"],
            result["unresolved_adjudication_count"] == 0,
        )
    )
    if json_output:
        atomic_json(result, json_output)
    return result
