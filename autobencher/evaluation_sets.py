"""Evaluation-set registry, role separation, and blind-set access control."""

from __future__ import annotations

import hashlib
import os
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

from .config import DEFAULT_TAXONOMY
from .dataset import template_signature


EVALUATION_ROLES = frozenset(
    {"development_regression", "official_fixed", "blind_final"}
)
BLIND_PATH_ENV = "AUTOBENCHER_BLIND_TEST_PATH"
BLIND_SHA256_ENV = "AUTOBENCHER_BLIND_TEST_SHA256"
BLIND_TOKEN_ENV = "AUTOBENCHER_BLIND_RELEASE_TOKEN"
BLIND_TOKEN_SHA256_ENV = "AUTOBENCHER_BLIND_RELEASE_TOKEN_SHA256"


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _resolve_path(value: str | Path, project_root: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(project_root).resolve() / path).resolve()


def load_evaluation_registry(
    registry_path: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    path = _resolve_path(registry_path, project_root)
    if not path.is_file():
        raise FileNotFoundError(f"Evaluation registry does not exist: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Evaluation registry must contain a mapping")
    raw_sets = payload.get("sets")
    if not isinstance(raw_sets, list) or not raw_sets:
        raise ValueError("Evaluation registry must contain a non-empty sets list")
    sets: dict[str, dict[str, Any]] = {}
    for raw in raw_sets:
        if not isinstance(raw, Mapping):
            raise ValueError("Every evaluation registry entry must be a mapping")
        identifier = str(raw.get("id", "")).strip()
        role = str(raw.get("role", "")).strip()
        if not identifier or identifier in sets:
            raise ValueError(f"Invalid or duplicate evaluation set id: {identifier!r}")
        if role not in EVALUATION_ROLES:
            raise ValueError(f"Invalid evaluation role for {identifier}: {role!r}")
        sets[identifier] = dict(raw)
    return {
        "schema_version": str(payload.get("schema_version", "1.0")),
        "path": path.as_posix(),
        "sha256": file_sha256(path),
        "sets": sets,
    }


def resolve_active_evaluation_set(
    config: Mapping[str, Any],
    project_root: str | Path,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Resolve one role and fail closed when a blind set is not released."""
    evaluation = config.get("evaluation_sets")
    if not isinstance(evaluation, Mapping):
        return {
            "id": "legacy_fixed_test",
            "role": "development_regression",
            "version": "legacy",
            "dataset_path": str(config["fixed_test"]["dataset_path"]),
            "minimum_questions_per_subcategory": 1,
            "require_all_subcategories": bool(
                config["fixed_test"].get("require_all_subcategories", False)
            ),
            "require_explicit_validation": False,
            "permissions": {
                "development_comparison": True,
                "method_selection": True,
                "hard_pool": False,
                "generator_prompt": False,
                "training": False,
                "threshold_tuning": True,
            },
            "registry_sha256": None,
        }
    registry = load_evaluation_registry(
        str(evaluation["registry_path"]),
        project_root,
    )
    active_id = str(evaluation["active_set"]).strip()
    if active_id not in registry["sets"]:
        raise ValueError(f"Unknown active evaluation set: {active_id!r}")
    selected = dict(registry["sets"][active_id])
    selected["registry_sha256"] = registry["sha256"]
    selected["registry_path"] = registry["path"]
    selected["phase"] = str(evaluation["phase"])

    if selected["role"] != "blind_final":
        if selected["role"] == "official_fixed":
            if str(evaluation["phase"]) not in {"frozen", "blind_evaluation"}:
                raise PermissionError(
                    "Official fixed evaluation is locked until the study "
                    "phase is frozen."
                )
            if bool(config.get("finetune", {}).get("enabled")):
                raise PermissionError(
                    "Fine-tuning is prohibited while official_fixed is active."
                )
            if bool(
                config.get("fixed_test", {}).get(
                    "evaluate_after_each_training_cycle",
                    False,
                )
            ):
                raise PermissionError(
                    "Official fixed scores cannot be evaluated after training "
                    "cycles or used for checkpoint/method selection."
                )
        selected_path = selected["dataset_path"]
        if (
            selected["role"] == "development_regression"
            and bool(selected.get("allow_fixed_test_path_override", False))
        ):
            selected_path = config["fixed_test"]["dataset_path"]
        selected["dataset_path"] = _resolve_path(
            str(selected_path),
            project_root,
        ).as_posix()
        return selected

    env = os.environ if environment is None else environment
    if str(evaluation["phase"]) != "blind_evaluation":
        raise PermissionError(
            "Blind evaluation is locked until evaluation_sets.phase is "
            "'blind_evaluation'."
        )
    if not bool(evaluation["allow_blind_evaluation"]):
        raise PermissionError(
            "Blind evaluation requires evaluation_sets.allow_blind_evaluation=true."
        )
    token_env = str(selected.get("release_token_env", BLIND_TOKEN_ENV))
    token_hash_env = str(
        selected.get("release_token_sha256_env", BLIND_TOKEN_SHA256_ENV)
    )
    path_env = str(selected.get("dataset_path_env", BLIND_PATH_ENV))
    hash_env = str(selected.get("dataset_sha256_env", BLIND_SHA256_ENV))
    token = str(env.get(token_env, "")).strip()
    expected_token_hash = str(env.get(token_hash_env, "")).strip().lower()
    if not token or not expected_token_hash:
        raise PermissionError(
            "Blind evaluation requires a release token and its independently "
            f"provisioned SHA-256 ({token_env}, {token_hash_env})"
        )
    if hashlib.sha256(token.encode("utf-8")).hexdigest() != expected_token_hash:
        raise PermissionError("Blind evaluation release token hash mismatch")
    raw_path = str(env.get(path_env, "")).strip()
    expected_sha256 = str(env.get(hash_env, "")).strip().lower()
    if not raw_path or not expected_sha256:
        raise PermissionError(
            f"Blind evaluation requires both {path_env} and {hash_env}."
        )
    path = Path(raw_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Released blind test set does not exist: {path}")
    observed_sha256 = file_sha256(path)
    if observed_sha256 != expected_sha256:
        raise PermissionError(
            "Blind test hash does not match the pre-registered release hash."
        )
    if bool(config.get("finetune", {}).get("enabled")):
        raise PermissionError("Fine-tuning is prohibited during blind evaluation.")
    selected["dataset_path"] = path.as_posix()
    selected["released_sha256"] = observed_sha256
    return selected


def validate_evaluation_coverage(
    questions: list[Mapping[str, Any]],
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate paper-set coverage and evidence without accepting silent gaps."""
    expected = {
        (category, subcategory)
        for category, subcategories in DEFAULT_TAXONOMY.items()
        for subcategory in subcategories
    }
    counts = Counter(
        (
            str(item.get("category", "")),
            str(item.get("sub_category", item.get("subcategory", ""))),
        )
        for item in questions
    )
    minimum = int(spec.get("minimum_questions_per_subcategory", 1))
    require_all = bool(spec.get("require_all_subcategories", True))
    insufficient = [
        {
            "category": category,
            "sub_category": subcategory,
            "count": counts[(category, subcategory)],
            "minimum": minimum,
        }
        for category, subcategory in sorted(expected)
        if require_all and counts[(category, subcategory)] < minimum
    ]
    if insufficient:
        detail = ", ".join(
            f"{item['category']}/{item['sub_category']}="
            f"{item['count']}<{item['minimum']}"
            for item in insufficient
        )
        raise ValueError(f"Evaluation set coverage is insufficient: {detail}")

    explicit = bool(spec.get("require_explicit_validation", False))
    invalid_evidence: list[str] = []
    status_counts: Counter[str] = Counter()
    diversity: dict[tuple[str, str], dict[str, set[Any]]] = {
        pair: {
            "difficulty_bands": set(),
            "answer_types": set(),
            "templates": set(),
            "reasoning_structures": set(),
        }
        for pair in expected
    }
    for item in questions:
        evidence = item.get("validation")
        if not isinstance(evidence, Mapping):
            evidence = item.get("verification")
        status = (
            str(evidence.get("status", "")).strip().lower()
            if isinstance(evidence, Mapping)
            else ""
        )
        if not status and isinstance(evidence, Mapping):
            status = (
                "verified"
                if bool(evidence.get("canonical_parse_passed"))
                else "unverified"
            )
        status_counts[status or "missing"] += 1
        pair = (
            str(item.get("category", "")),
            str(item.get("sub_category", item.get("subcategory", ""))),
        )
        if pair in diversity:
            difficulty = int(item.get("difficulty", 5))
            band = "basic" if difficulty <= 3 else (
                "intermediate" if difficulty <= 6 else "hard"
            )
            diversity[pair]["difficulty_bands"].add(band)
            diversity[pair]["answer_types"].add(
                str(item.get("answer_type", ""))
            )
            diversity[pair]["templates"].add(
                template_signature(item.get("question", ""))
            )
            diversity[pair]["reasoning_structures"].add(
                str(item.get("reasoning_structure", "")).strip()
            )
        if explicit:
            sources = (
                evidence.get("sources", [])
                if isinstance(evidence, Mapping)
                else []
            )
            source_ids = [
                str(source.get("source_id", "")).strip()
                for source in sources if isinstance(source, Mapping)
            ] if isinstance(sources, list) else []
            methods = [
                str(source.get("method", source.get("source_type", ""))).strip()
                for source in sources if isinstance(source, Mapping)
            ] if isinstance(sources, list) else []
            actors = [
                str(
                    source.get(
                        "solver_id",
                        source.get("model_id", source.get("annotator_id", "")),
                    )
                ).strip()
                for source in sources if isinstance(source, Mapping)
            ] if isinstance(sources, list) else []
            answers = [
                " ".join(str(source.get("answer", "")).split()).lower()
                for source in sources if isinstance(source, Mapping)
            ] if isinstance(sources, list) else []
            gold = " ".join(
                str(item.get("canonical_answer", item.get("gold_answer", ""))).split()
            ).lower()
            independent_sources = (
                len(source_ids) >= 2
                and all(source_ids)
                and len(set(source_ids)) == len(source_ids)
                and all(methods)
                and len(set(methods)) >= 2
                and all(actors)
                and len(set(actors)) == len(actors)
                and all(answers)
                and all(answer == gold for answer in answers)
            )
            if status != "verified" or not independent_sources:
                invalid_evidence.append(
                    str(item.get("question_id", item.get("id", "<unknown>")))
                )
                continue
            subcategory = str(
                item.get("sub_category", item.get("subcategory", ""))
            )
            if (
                subcategory == "Proof and Mathematical Reasoning"
                or str(item.get("answer_type", "")) == "text"
            ):
                annotations = evidence.get("human_annotations", [])
                annotators = {
                    str(annotation.get("annotator_id", "")).strip()
                    for annotation in annotations
                    if isinstance(annotation, Mapping)
                    and str(annotation.get("label", "")).strip()
                }
                labels = {
                    str(annotation.get("label", "")).strip()
                    for annotation in annotations
                    if isinstance(annotation, Mapping)
                    and str(annotation.get("label", "")).strip()
                }
                adjudication = evidence.get("adjudication")
                if len(annotators) < 2 or (
                    len(labels) > 1
                    and not (
                        isinstance(adjudication, Mapping)
                        and str(adjudication.get("label", "")).strip()
                        and str(
                            adjudication.get("adjudicator_id", "")
                        ).strip()
                    )
                ):
                    invalid_evidence.append(
                        str(
                            item.get(
                                "question_id",
                                item.get("id", "<unknown>"),
                            )
                        )
                    )
    if invalid_evidence:
        raise ValueError(
            "Official/blind questions require verified status, distinct non-empty "
            "source and solver/model/annotator IDs, at least two methods, and "
            "answers agreeing with gold; "
            "proof/text items also require two human "
            "annotations and adjudication on disagreement: "
            + ", ".join(invalid_evidence[:20])
        )
    diversity_requirements = {
        "difficulty_bands": int(spec.get("minimum_difficulty_bands", 0)),
        "answer_types": int(spec.get("minimum_answer_types", 0)),
        "templates": int(spec.get("minimum_template_clusters", 0)),
        "reasoning_structures": int(
            spec.get("minimum_reasoning_structures", 0)
        ),
    }
    diversity_failures = []
    if explicit:
        for pair, observed_values in sorted(diversity.items()):
            for field, minimum_value in diversity_requirements.items():
                values = {
                    value for value in observed_values[field] if value
                }
                if len(values) < minimum_value:
                    diversity_failures.append(
                        f"{pair[0]}/{pair[1]}:{field}="
                        f"{len(values)}<{minimum_value}"
                    )
    if diversity_failures:
        raise ValueError(
            "Evaluation set diversity is insufficient: "
            + ", ".join(diversity_failures[:30])
        )
    return {
        "minimum_questions_per_subcategory": minimum,
        "subcategory_counts": {
            f"{category} / {subcategory}": counts[(category, subcategory)]
            for category, subcategory in sorted(expected)
        },
        "validation_status_counts": dict(sorted(status_counts.items())),
        "diversity_requirements": diversity_requirements,
        "diversity_counts": {
            f"{category} / {subcategory}": {
                field: len({value for value in values if value})
                for field, values in observed.items()
            }
            for (category, subcategory), observed in sorted(diversity.items())
        },
    }
