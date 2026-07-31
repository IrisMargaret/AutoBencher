"""Human-review exports and error-attribution agreement metrics."""

from __future__ import annotations

import csv
import json
import math
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .experiment import atomic_json, utc_now
from .fingerprints import file_sha256
from .structured import ERROR_TAGS


ERROR_CODEBOOK = {
    "arithmetic_computation_error": "A valid arithmetic operation is evaluated incorrectly.",
    "sign_error": "The earliest error reverses, drops, or introduces a sign.",
    "reciprocal_error": "A quantity is incorrectly inverted or a division ratio is reversed.",
    "scale_or_percentage_error": "A factor, rate, percentage, or unit scale is applied incorrectly.",
    "rounding_error": "An explicitly requested rounding rule is applied incorrectly.",
    "numeric_approximation_error": "Correct exact reasoning is followed by an inaccurate irrational-number approximation.",
    "off_by_one_error": "A count or endpoint differs by exactly one because a boundary is mishandled.",
    "answer_transfer_error": "The reasoning reaches the correct result but the final-answer field copies a different value.",
    "constraint_violation": "The proposed result fails an explicit domain, range, or problem constraint.",
    "incomplete_solution": "The response stops before satisfying every requested component.",
    "unit_mismatch": "The numeric result is paired with an incompatible or missing required unit.",
    "symbolic_manipulation_error": "The earliest algebraic or symbolic transformation is not equivalent.",
    "invalid_multiple_choice": "The response selects an option not available in the question.",
    "concept_confusion": "A named mathematical concept is applied as a different concept; use only with explicit evidence.",
    "formula_memory_error": "An explicit formula is stated incorrectly before otherwise consistent substitution.",
    "calculation_error": "A calculation is wrong but the available trace cannot support a more specific numeric tag.",
    "multi_step_logic_error": "The dependency between otherwise interpretable steps is invalid.",
    "condition_missing": "A required case, hypothesis, or solution condition is omitted.",
    "format_output_error": "The mathematical content is recoverable but violates the required output contract.",
    "tool_violation": "The response invokes or exposes a prohibited external tool or tool trace.",
    "irrelevant_output": "The response is dominated by material unrelated to solving the question.",
    "prompt_echo": "The response copies role instructions or the prompt instead of answering.",
    "parse_failed": "The response cannot be parsed under the declared answer/output schema.",
    "unknown_error": "Available evidence cannot isolate one reliable earliest causal mechanism.",
}


def _confidence_band(record: Mapping[str, Any]) -> str:
    try:
        confidence = float(record.get("attribution_confidence", 0.0))
    except (TypeError, ValueError):
        return "missing"
    if confidence >= 0.8:
        return "high"
    if confidence >= 0.5:
        return "medium"
    return "low"


def _difficulty_band(record: Mapping[str, Any]) -> str:
    try:
        difficulty = float(record.get("difficulty", 0))
    except (TypeError, ValueError):
        return "unknown"
    if difficulty <= 3:
        return "easy"
    if difficulty <= 6:
        return "medium"
    return "hard"


def _review_stratum(record: Mapping[str, Any]) -> tuple[str, ...]:
    return (
        str(record.get("primary_error_tag") or "unknown_error"),
        _confidence_band(record),
        str(record.get("verification_tier") or "unknown"),
        str(record.get("category") or "unknown"),
        _difficulty_band(record),
    )


def _stratified_review_sample(
    records: list[dict[str, Any]], sample_size: int, seed: int
) -> list[dict[str, Any]]:
    """Deterministically cover rare error/confidence/category strata first."""
    strata: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        strata[_review_stratum(record)].append(record)
    rng = random.Random(seed)
    for key in strata:
        rng.shuffle(strata[key])
    ordered_keys = sorted(strata, key=lambda key: (len(strata[key]), key))
    selected: list[dict[str, Any]] = []
    selected_by_stratum: Counter[tuple[str, ...]] = Counter()
    while len(selected) < min(sample_size, len(records)):
        progressed = False
        for key in ordered_keys:
            position = selected_by_stratum[key]
            if position < len(strata[key]):
                selected.append(strata[key][position])
                selected_by_stratum[key] += 1
                progressed = True
                if len(selected) >= min(sample_size, len(records)):
                    break
        if not progressed:
            break
    for record in selected:
        key = _review_stratum(record)
        probability = selected_by_stratum[key] / len(strata[key])
        record["_review_sampling"] = {
            "stratum": list(key),
            "population_count": len(strata[key]),
            "sample_count": selected_by_stratum[key],
            "selection_probability": probability,
            "sample_weight": 1.0 / probability,
        }
    return selected


def _write_error_codebook(path: Path) -> None:
    definitions = {label: ERROR_CODEBOOK[label] for label in ERROR_TAGS}
    atomic_json(
        {
            "schema_version": "1.0",
            "allowed_labels": list(ERROR_TAGS),
            "definitions": definitions,
            "instruction": (
                "Assign the earliest causal error visible in the raw response; "
                "do not infer hidden evaluator or solver evidence."
            ),
        },
        path,
    )


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
    sealed_output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Export two independent blind packets and a sealed system prediction file."""

    errors = [dict(record) for record in records if not record.get("is_correct")]
    selected = _stratified_review_sample(errors, sample_size, seed)
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    public_fields = [
        "question_id",
        "question",
        "gold_answer",
        "test_taker_response",
        "raw_reasoning",
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
        "review_stratum",
        "selection_probability",
        "sample_weight",
    ]
    sealed_root = (
        Path(sealed_output_dir)
        if sealed_output_dir is not None
        else root.parent / f".{root.name}_sealed"
    )
    sealed_root.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(sealed_root, 0o700)
    except OSError:
        pass
    system_path = sealed_root / "system_predictions.csv"
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
                    "review_stratum": json.dumps(
                        record["_review_sampling"]["stratum"],
                        ensure_ascii=False,
                    ),
                    "selection_probability": record["_review_sampling"][
                        "selection_probability"
                    ],
                    "sample_weight": record["_review_sampling"]["sample_weight"],
                }
            )
    try:
        os.chmod(system_path, 0o600)
    except OSError:
        pass
    codebook_path = root / "error_taxonomy_codebook.json"
    _write_error_codebook(codebook_path)
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
                        "raw_reasoning": record.get(
                            "raw_reasoning",
                            record.get("test_taker_reasoning", ""),
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
            "system_predictions_are_outside_public_directory": True,
        },
        "files": {
            "system_predictions": {
                # Never disclose the sealed path in the reviewer-facing manifest.
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
            "codebook": {
                "path": codebook_path.name,
                "sha256": file_sha256(codebook_path),
            },
        },
        "sampling": {
            "method": "rare_stratum_round_robin",
            "stratum_fields": [
                "system_error_tag",
                "confidence_band",
                "verification_tier",
                "category",
                "difficulty_band",
            ],
            "inverse_probability_weights_stored_in_sealed_file": True,
        },
    }
    atomic_json(manifest, root / "review_manifest.json")
    # The private locator is returned to the operator but is deliberately not
    # serialized into review_manifest.json or either annotator packet.
    return {**manifest, "operator_sealed_system_path": system_path.as_posix()}


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
        "review_stratum",
        "selection_probability",
        "sample_weight",
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
                    "review_stratum": source.get("review_stratum", "[]"),
                    "selection_probability": source.get(
                        "selection_probability", ""
                    ),
                    "sample_weight": source.get("sample_weight", "1"),
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


def _sample_weight(row: Mapping[str, Any]) -> float:
    try:
        value = float(row.get("sample_weight", 1.0))
    except (TypeError, ValueError):
        value = 1.0
    return value if math.isfinite(value) and value > 0 else 1.0


def _cohen_kappa(
    left: list[str],
    right: list[str],
    weights: list[float] | None = None,
) -> float | None:
    raw_weights = weights or [1.0] * len(left)
    pairs = [
        (a, b, weight)
        for a, b, weight in zip(left, right, raw_weights)
        if a and b
    ]
    if not pairs:
        return None
    labels = sorted({value for a, b, _ in pairs for value in (a, b)})
    total_weight = sum(weight for _, _, weight in pairs)
    observed = sum(weight * (a == b) for a, b, weight in pairs) / total_weight
    left_counts = Counter()
    right_counts = Counter()
    for a, b, weight in pairs:
        left_counts[a] += weight
        right_counts[b] += weight
    expected = sum(
        left_counts[label] / total_weight * right_counts[label] / total_weight
        for label in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def evaluate_review_csv(
    csv_path: str | Path,
    json_output: str | Path | None = None,
    *,
    minimum_reviewed_count: int = 300,
    minimum_cohen_kappa: float = 0.70,
    minimum_high_confidence_accuracy: float = 0.80,
    minimum_high_confidence_count: int = 50,
    minimum_completion_rate: float = 0.90,
    maximum_unresolved_rate: float = 0.0,
    minimum_per_label_count: int = 5,
) -> dict[str, Any]:
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    allowed_labels = set(ERROR_TAGS)
    observed_human_labels = {
        value.strip()
        for row in rows
        for value in (
            str(row.get("human_label_1", "")),
            str(row.get("human_label_2", "")),
            str(row.get("adjudicated_label", "")),
        )
        if value.strip()
    }
    invalid_labels = sorted(observed_human_labels - allowed_labels)
    if invalid_labels:
        raise ValueError(
            f"Review contains labels outside the frozen taxonomy: {invalid_labels}"
        )
    invalid_predictions = sorted(
        {
            str(row.get("primary_error_tag", "")).strip()
            for row in rows
            if str(row.get("primary_error_tag", "")).strip()
        }
        - allowed_labels
    )
    if invalid_predictions:
        raise ValueError(
            f"System predictions contain unknown labels: {invalid_predictions}"
        )
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
            confusion[actual][predicted] += _sample_weight(row)
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
            [_sample_weight(row) for row in rows],
        ),
    }
    reviewed_pairs = [
        row
        for row in rows
        if row.get("primary_error_tag") and _gold_label(row)
    ]
    weighted_review_total = sum(_sample_weight(row) for row in reviewed_pairs)
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
            (
                float(row.get("primary_error_tag") == _gold_label(row)),
                _sample_weight(row),
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
        calibration_pairs.append((confidence, correct, _sample_weight(row)))
        if confidence >= 0.8:
            high_confidence.append((confidence, correct, _sample_weight(row)))
    result.update(
        {
            # Ragas-style composable quality signals: correctness, coverage,
            # evidence completeness, and calibration are reported separately.
            "attribution_accuracy": (
                sum(
                    _sample_weight(row)
                    * (row.get("primary_error_tag") == _gold_label(row))
                    for row in reviewed_pairs
                )
                / weighted_review_total
                if reviewed_pairs
                else None
            ),
            "selective_coverage": (
                sum(_sample_weight(row) for row in non_abstained)
                / weighted_review_total
                if reviewed_pairs
                else None
            ),
            "selective_accuracy": (
                sum(
                    _sample_weight(row)
                    * (row.get("primary_error_tag") == _gold_label(row))
                    for row in non_abstained
                )
                / sum(_sample_weight(row) for row in non_abstained)
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
                sum(
                    _sample_weight(row) * bool(_evidence_items(row))
                    for row in reviewed_pairs
                )
                / weighted_review_total
                if reviewed_pairs
                else None
            ),
            "verified_evidence_coverage": (
                sum(
                    _sample_weight(row) * any(
                        bool(item.get("check_name"))
                        for item in _evidence_items(row)
                    )
                    for row in reviewed_pairs
                )
                / weighted_review_total
                if reviewed_pairs
                else None
            ),
            "confidence_brier_score": (
                sum(
                    weight * (confidence - correct) ** 2
                    for confidence, correct, weight in calibration_pairs
                )
                / sum(weight for _, _, weight in calibration_pairs)
                if calibration_pairs
                else None
            ),
            "high_confidence_accuracy": (
                sum(correct * weight for _, correct, weight in high_confidence)
                / sum(weight for _, _, weight in high_confidence)
                if high_confidence
                else None
            ),
            "high_confidence_count": len(high_confidence),
            "unknown_error_ratio": (
                sum(
                    _sample_weight(row)
                    * (row.get("primary_error_tag") == "unknown_error")
                    for row in reviewed_pairs
                )
                / weighted_review_total
                if reviewed_pairs
                else None
            ),
            "unknown_error_policy": (
                "unknown_error is a deliberate abstention when deterministic "
                "evidence cannot isolate one mechanism; report its ratio "
                "instead of forcing a low-confidence class."
            ),
            "unresolved_adjudication_count": sum(
                not _gold_label(row)
                for row in rows
            ),
            "accuracy_by_verification_tier": {
                tier: (
                    sum(correct * weight for correct, weight in values)
                    / sum(weight for _, weight in values)
                )
                for tier, values in sorted(accuracy_by_tier.items())
            },
        }
    )
    kappa = result["cohen_kappa"]
    high_accuracy = result["high_confidence_accuracy"]
    completion_rate = result["reviewed_count"] / len(rows) if rows else 0.0
    unresolved_rate = (
        result["unresolved_adjudication_count"] / len(rows) if rows else 1.0
    )
    gold_label_counts = Counter(_gold_label(row) for row in rows if _gold_label(row))
    underrepresented_labels = {
        label: count
        for label, count in sorted(gold_label_counts.items())
        if count < minimum_per_label_count
    }
    result["acceptance"] = {
        "minimum_reviewed_count": minimum_reviewed_count,
        "minimum_cohen_kappa": minimum_cohen_kappa,
        "minimum_high_confidence_accuracy": minimum_high_confidence_accuracy,
        "minimum_high_confidence_count": minimum_high_confidence_count,
        "minimum_completion_rate": minimum_completion_rate,
        "maximum_unresolved_rate": maximum_unresolved_rate,
        "minimum_per_label_count": minimum_per_label_count,
        "completion_rate": completion_rate,
        "unresolved_rate": unresolved_rate,
        "underrepresented_observed_labels": underrepresented_labels,
        "reviewed_count_pass": result["reviewed_count"] >= minimum_reviewed_count,
        "cohen_kappa_pass": (
            kappa is not None and kappa >= minimum_cohen_kappa
        ),
        "high_confidence_accuracy_pass": (
            high_accuracy is not None
            and high_accuracy >= minimum_high_confidence_accuracy
        ),
        "high_confidence_count_pass": (
            result["high_confidence_count"] >= minimum_high_confidence_count
        ),
        "completion_rate_pass": completion_rate >= minimum_completion_rate,
        "unresolved_rate_pass": unresolved_rate <= maximum_unresolved_rate,
        "per_observed_label_count_pass": not underrepresented_labels,
    }
    result["acceptance"]["paper_core_signal_ready"] = all(
        (
            result["acceptance"]["reviewed_count_pass"],
            result["acceptance"]["cohen_kappa_pass"],
            result["acceptance"]["high_confidence_accuracy_pass"],
            result["acceptance"]["high_confidence_count_pass"],
            result["acceptance"]["completion_rate_pass"],
            result["acceptance"]["unresolved_rate_pass"],
            result["acceptance"]["per_observed_label_count_pass"],
        )
    )
    if json_output:
        atomic_json(result, json_output)
    return result
