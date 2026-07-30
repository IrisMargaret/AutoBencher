"""Loading, validating, and summarizing the immutable fixed math holdout."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import DEFAULT_TAXONOMY
from .structured import ANSWER_TYPES, normalize_answer_type


def resolve_fixed_test_path(
    config: Mapping[str, Any],
    project_root: str | Path | None = None,
) -> Path:
    configured = Path(str(config["fixed_test"]["dataset_path"])).expanduser()
    if configured.is_absolute():
        return configured
    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )
    candidates = [Path.cwd() / configured, root / configured]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[-1].resolve()


def fixed_test_sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_fixed_test_set(
    config: Mapping[str, Any],
    project_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load the fixed set and fail closed on taxonomy or schema drift."""
    path = resolve_fixed_test_path(config, project_root)
    if not path.is_file():
        raise FileNotFoundError(f"Fixed test set does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Fixed test set is invalid JSON: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("questions"), list
    ):
        raise ValueError("Fixed test set must contain a questions array")
    required = {
        "question_id",
        "category",
        "sub_category",
        "difficulty",
        "question",
        "answer_type",
        "canonical_answer",
    }
    questions = []
    identifiers = set()
    normalized_questions = set()
    for index, source in enumerate(payload["questions"]):
        if not isinstance(source, dict) or not required.issubset(source):
            raise ValueError(
                f"Fixed test question {index} is missing required fields"
            )
        record = dict(source)
        identifier = str(record["question_id"]).strip()
        question_text = " ".join(str(record["question"]).split())
        if not identifier or identifier in identifiers:
            raise ValueError(
                f"Fixed test question_id is empty or duplicated: {identifier!r}"
            )
        if not question_text or question_text.lower() in normalized_questions:
            raise ValueError(
                f"Fixed test question is empty or duplicated: {question_text!r}"
            )
        category = str(record["category"])
        subcategory = str(record["sub_category"])
        if (
            category not in DEFAULT_TAXONOMY
            or subcategory not in DEFAULT_TAXONOMY[category]
        ):
            raise ValueError(
                f"Fixed test taxonomy entry is invalid: "
                f"{category} / {subcategory}"
            )
        answer_type = normalize_answer_type(
            record["answer_type"],
            record["canonical_answer"],
        )
        if answer_type not in ANSWER_TYPES:
            raise ValueError(
                f"Fixed test answer type is invalid: {answer_type}"
            )
        difficulty = int(record["difficulty"])
        if not 1 <= difficulty <= 10:
            raise ValueError("Fixed test difficulty must be within [1, 10]")
        identifiers.add(identifier)
        normalized_questions.add(question_text.lower())
        questions.append(
            {
                **record,
                "id": identifier,
                "question_id": identifier,
                "question": question_text,
                "subcategory": subcategory,
                "sub_category": subcategory,
                "answer_type": answer_type,
                "canonical_answer": str(
                    record["canonical_answer"]
                ).strip(),
                "gold_answer": str(record["canonical_answer"]).strip(),
                "answer": str(record["canonical_answer"]).strip(),
                "display_answer": str(record["canonical_answer"]).strip(),
                "difficulty": difficulty,
                "fixed_test": True,
                "truth_validation_details": {
                    "source_question_sha256": hashlib.sha256(
                        question_text.encode("utf-8")
                    ).hexdigest(),
                    "solver_question_sha256": hashlib.sha256(
                        question_text.encode("utf-8")
                    ).hexdigest(),
                    "solver_branch": "checked_in_fixed_holdout",
                    "substitution_passed": True,
                },
            }
        )
    expected = {
        (category, subcategory)
        for category, subcategories in DEFAULT_TAXONOMY.items()
        for subcategory in subcategories
    }
    observed = {
        (record["category"], record["sub_category"])
        for record in questions
    }
    missing = sorted(expected - observed)
    if config["fixed_test"]["require_all_subcategories"] and missing:
        raise ValueError(
            "Fixed test set does not cover every subcategory: "
            + ", ".join(f"{a}/{b}" for a, b in missing)
        )
    metadata = {
        "schema_version": str(payload.get("schema_version", "1.0")),
        "name": str(payload.get("name", path.stem)),
        "description": str(payload.get("description", "")),
        "builder_manifest_sha256": payload.get(
            "builder_manifest_sha256"
        ),
        "path": str(path).replace("\\", "/"),
        "sha256": fixed_test_sha256(path),
        "question_count": len(questions),
        "covered_subcategory_count": len(observed),
        "required_subcategory_count": len(expected),
        "missing_subcategories": [
            {"category": category, "sub_category": subcategory}
            for category, subcategory in missing
        ],
        "source_counts": dict(
            sorted(
                Counter(
                    str(record.get("source_dataset", "project_native"))
                    for record in questions
                ).items()
            )
        ),
        "source_manifests": list(payload.get("sources", [])),
        "selection_policy": dict(payload.get("selection_policy", {})),
    }
    return questions, metadata


def fixed_benchmark_summary(
    records: Iterable[Mapping[str, Any]],
    *,
    stage: str,
    model_name: str,
    dataset_sha256: str,
) -> dict[str, Any]:
    records = [dict(record) for record in records]
    total = len(records)
    correct = sum(bool(record.get("is_correct")) for record in records)
    answer_type_groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    subcategory_groups: dict[tuple[str, str], dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    source_groups: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    parsed_confidences = []
    reasoning_record_count = 0
    semantic_judge_success_count = 0
    for record in records:
        answer_type = str(record.get("answer_type", "text"))
        answer_type_groups[answer_type]["total"] += 1
        answer_type_groups[answer_type]["correct"] += int(
            bool(record.get("is_correct"))
        )
        category = str(record.get("category", ""))
        subcategory = str(
            record.get("sub_category", record.get("subcategory", ""))
        )
        group = subcategory_groups[(category, subcategory)]
        group["total"] += 1
        group["correct"] += int(bool(record.get("is_correct")))
        source = str(record.get("source_dataset", "project_native"))
        source_groups[source]["total"] += 1
        source_groups[source]["correct"] += int(
            bool(record.get("is_correct"))
        )
        parsed = record.get("parsed_response")
        if isinstance(parsed, Mapping):
            reasoning = parsed.get("reasoning_summary")
            if isinstance(reasoning, list) and any(
                str(step).strip() for step in reasoning
            ):
                reasoning_record_count += 1
            try:
                confidence = float(parsed.get("confidence"))
            except (TypeError, ValueError):
                confidence = None
            if confidence is not None and 0 <= confidence <= 1:
                parsed_confidences.append(confidence)
        semantic = record.get("semantic_judge")
        if (
            isinstance(semantic, Mapping)
            and semantic.get("status") == "success"
        ):
            semantic_judge_success_count += 1

    answer_type_statistics = [
        {
            "answer_type": answer_type,
            **counts,
            "accuracy": (
                counts["correct"] / counts["total"]
                if counts["total"]
                else 0.0
            ),
        }
        for answer_type, counts in sorted(answer_type_groups.items())
    ]
    subcategory_statistics = [
        {
            "category": category,
            "sub_category": subcategory,
            **counts,
            "accuracy": (
                counts["correct"] / counts["total"]
                if counts["total"]
                else 0.0
            ),
        }
        for (category, subcategory), counts in sorted(
            subcategory_groups.items()
        )
    ]
    source_statistics = [
        {
            "source_dataset": source,
            **counts,
            "accuracy": (
                counts["correct"] / counts["total"]
                if counts["total"]
                else 0.0
            ),
        }
        for source, counts in sorted(source_groups.items())
    ]
    return {
        "stage": str(stage),
        "model_name": str(model_name).replace("\\", "/"),
        "dataset_sha256": str(dataset_sha256),
        "total_questions": total,
        "correct_questions": correct,
        "accuracy": correct / total if total else 0.0,
        "parse_failure_count": sum(
            record.get("parse_status") != "success" for record in records
        ),
        "tool_violation_count": sum(
            bool(record.get("tool_violation")) for record in records
        ),
        "reasoning_record_count": reasoning_record_count,
        "semantic_judge_success_count": semantic_judge_success_count,
        "format_only_error_count": sum(
            bool(record.get("format_only_error")) for record in records
        ),
        "mean_test_taker_confidence": (
            sum(parsed_confidences) / len(parsed_confidences)
            if parsed_confidences
            else None
        ),
        "answer_type_statistics": answer_type_statistics,
        "subcategory_statistics": subcategory_statistics,
        "source_statistics": source_statistics,
    }
