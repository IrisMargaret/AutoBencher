"""Loading, validating, and summarizing the immutable fixed math holdout."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import DEFAULT_TAXONOMY
from .difficulty import analyze_difficulty
from .experiment import atomic_json
from .structured import (
    ANSWER_TYPES,
    normalize_answer_type,
    normalize_generated_gold_contract,
)


PROJECT_FIXED_TEST_SET = (
    Path(__file__).resolve().parents[1]
    / "benchmarks"
    / "fixed_math_test_set.json"
)
EXCLUDED_EXTERNAL_QUESTION_SOURCES = frozenset(
    {
        "openai/gsm8k",
        "eleutherai/hendrycks_math",
        "cais/mmlu",
    }
)


def _assert_project_native_questions(
    questions: Iterable[Mapping[str, Any]],
) -> None:
    """Reject records carrying provenance from removed external question sets."""
    excluded = []
    for record in questions:
        source = str(record.get("source_dataset", "project_native")).strip()
        if source.lower() in EXCLUDED_EXTERNAL_QUESTION_SOURCES:
            excluded.append(
                f"{record.get('question_id', '<unknown>')}={source}"
            )
    if excluded:
        raise ValueError(
            "Fixed test set contains excluded Hugging Face question sources: "
            + ", ".join(excluded)
        )


def install_project_fixed_test_set(
    output: str | Path,
    *,
    allowed_data_root: str | Path,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Install the checked-in, project-native holdout without network access."""
    output_path = Path(output).expanduser().resolve()
    allowed_root = Path(allowed_data_root).expanduser().resolve()
    if not output_path.is_relative_to(allowed_root):
        raise ValueError(
            f"Output must be beneath allowed data root {allowed_root}"
        )

    validation_config = {
        "fixed_test": {
            "dataset_path": PROJECT_FIXED_TEST_SET.as_posix(),
            "require_all_subcategories": True,
        }
    }
    questions, source_metadata = load_fixed_test_set(validation_config)
    _assert_project_native_questions(questions)
    source_bytes = PROJECT_FIXED_TEST_SET.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()

    if output_path.is_file():
        existing_sha256 = fixed_test_sha256(output_path)
        if existing_sha256 == source_sha256:
            status = "already_installed"
        elif not overwrite:
            raise FileExistsError(
                f"Refusing to replace different fixed test set: {output_path}. "
                "Use --overwrite only when intentionally starting a new "
                "benchmark version."
            )
        else:
            status = "replaced"
    else:
        status = "installed"

    if status != "already_installed":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = output_path.with_name(output_path.name + ".tmp")
        temporary.write_bytes(source_bytes)
        os.replace(temporary, output_path)

    manifest = {
        "schema_version": "1.0",
        "name": source_metadata["name"],
        "question_source": "project_native",
        "network_access": False,
        "excluded_question_sources": sorted(
            EXCLUDED_EXTERNAL_QUESTION_SOURCES
        ),
        "output_path": output_path.as_posix(),
        "sha256": source_sha256,
        "question_count": source_metadata["question_count"],
        "covered_subcategory_count": source_metadata[
            "covered_subcategory_count"
        ],
        "required_subcategory_count": source_metadata[
            "required_subcategory_count"
        ],
    }
    atomic_json(manifest, output_path.with_suffix(output_path.suffix + ".manifest.json"))
    return {"status": status, **manifest}


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
        answer_contract = normalize_generated_gold_contract(
            question_text,
            record["canonical_answer"],
            answer_type,
            record.get("tolerance"),
        )
        answer_type = answer_contract["answer_type"]
        record.update(
            {
                key: value
                for key, value in answer_contract.items()
                if value is not None or key != "exact_canonical_answer"
            }
        )
        if answer_type not in ANSWER_TYPES:
            raise ValueError(
                f"Fixed test answer type is invalid: {answer_type}"
            )
        difficulty = int(record["difficulty"])
        if not 1 <= difficulty <= 10:
            raise ValueError("Fixed test difficulty must be within [1, 10]")
        difficulty_profile = record.get("difficulty_profile")
        if not isinstance(difficulty_profile, Mapping):
            difficulty_profile = analyze_difficulty(
                question_text,
                answer_type,
            )
            difficulty_profile.update(
                {
                    "declared_source_score": difficulty,
                    "effective_score": difficulty,
                    "profile_role": (
                        "legacy_fixed_set_diagnostic_only"
                    ),
                }
            )
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
                    answer_contract["canonical_answer"]
                ).strip(),
                "gold_answer": str(
                    answer_contract["canonical_answer"]
                ).strip(),
                "answer": str(
                    answer_contract["canonical_answer"]
                ).strip(),
                "display_answer": str(
                    answer_contract["display_answer"]
                ).strip(),
                "tolerance": answer_contract["tolerance"],
                "difficulty": difficulty,
                "target_difficulty": int(
                    record.get("target_difficulty", difficulty)
                ),
                "observed_difficulty": int(
                    record.get(
                        "observed_difficulty",
                        difficulty_profile["score"],
                    )
                ),
                "difficulty_profile": dict(difficulty_profile),
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
        "difficulty_rubric_versions": sorted(
            {
                str(
                    record.get("difficulty_profile", {}).get(
                        "rubric_version",
                        "unknown",
                    )
                )
                for record in questions
            }
        ),
    }
    _assert_project_native_questions(questions)
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
    difficulty_groups: dict[int, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "correct": 0}
    )
    difficulty_gaps = []
    dimension_values: dict[str, list[float]] = defaultdict(list)
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
        difficulty = int(record.get("difficulty", 5))
        difficulty_groups[difficulty]["total"] += 1
        difficulty_groups[difficulty]["correct"] += int(
            bool(record.get("is_correct"))
        )
        profile = record.get("difficulty_profile")
        if isinstance(profile, Mapping):
            requested = profile.get(
                "requested_score",
                record.get("target_difficulty"),
            )
            observed = profile.get(
                "score",
                record.get("observed_difficulty"),
            )
            if requested is not None and observed is not None:
                difficulty_gaps.append(abs(float(observed) - float(requested)))
            dimensions = profile.get("dimensions", {})
            if isinstance(dimensions, Mapping):
                for name, dimension in dimensions.items():
                    if isinstance(dimension, Mapping):
                        try:
                            dimension_values[str(name)].append(
                                float(dimension["value"])
                            )
                        except (KeyError, TypeError, ValueError):
                            pass
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
    difficulty_statistics = [
        {
            "difficulty": difficulty,
            **counts,
            "accuracy": (
                counts["correct"] / counts["total"]
                if counts["total"]
                else 0.0
            ),
        }
        for difficulty, counts in sorted(difficulty_groups.items())
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
        "difficulty_statistics": difficulty_statistics,
        "difficulty_calibration": {
            "profiled_count": sum(
                1
                for record in records
                if isinstance(record.get("difficulty_profile"), Mapping)
            ),
            "mean_absolute_target_gap": (
                sum(difficulty_gaps) / len(difficulty_gaps)
                if difficulty_gaps
                else None
            ),
            "mean_dimension_values": {
                name: sum(values) / len(values)
                for name, values in sorted(dimension_values.items())
                if values
            },
        },
    }
