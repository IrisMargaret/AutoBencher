"""Out-of-band construction of a contamination-checked blind benchmark."""

from __future__ import annotations

import hashlib
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from autobencher.dataset import normalize_question_text, template_signature, token_jaccard
from autobencher.evaluation_audit import math_ast_signature, math_structure_signature
from autobencher.evaluation_sets import validate_evaluation_coverage
from autobencher.experiment import atomic_json


def _load_questions(path: str | Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, Mapping):
        payload = payload.get("questions", payload.get("data"))
    if not isinstance(payload, list):
        raise ValueError(f"Question file must contain a list: {path}")
    return [dict(item) for item in payload]


def _collision(candidate: Mapping[str, Any], prior: Mapping[str, Any]) -> str | None:
    candidate_text = candidate.get("question", "")
    prior_text = prior.get("question", "")
    if normalize_question_text(candidate_text) == normalize_question_text(prior_text):
        return "exact_text"
    candidate_template = template_signature(candidate_text)
    if candidate_template and candidate_template == template_signature(prior_text):
        return "parameterized_template"
    candidate_ast = math_ast_signature(candidate_text)
    if candidate_ast and candidate_ast == math_ast_signature(prior_text):
        if math_structure_signature(candidate_text) == math_structure_signature(prior_text):
            return "math_structure"
    candidate_structure = math_structure_signature(candidate_text)
    prior_structure = math_structure_signature(prior_text)
    candidate_tokens = set(candidate_structure.split("|")) - {""}
    prior_tokens = set(prior_structure.split("|")) - {""}
    structural_overlap = (
        len(candidate_tokens & prior_tokens)
        / max(1, len(candidate_tokens | prior_tokens))
    )
    operation_markers = {
        "differentiate",
        "integrate",
        "derivative",
        "integral",
        "matrix",
        "probability",
        "solve",
    }
    if (
        structural_overlap >= 0.8
        and candidate_tokens & prior_tokens & operation_markers
    ):
        return "math_structure"
    if max(
        token_jaccard(candidate_text, prior_text, ngram=1),
        token_jaccard(candidate_text, prior_text, ngram=2),
    ) >= 0.85:
        return "near_text"
    return None


def prepare_blind_benchmark(
    *,
    candidate_path: str | Path,
    contamination_paths: Iterable[str | Path],
    seed_file: str | Path,
    output_path: str | Path,
    manifest_path: str | Path,
    allowed_data_root: str | Path,
    questions_per_subcategory: int = 20,
) -> dict[str, Any]:
    """Select once, fail closed, and never expose seed or item content."""
    output = Path(output_path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    allowed = Path(allowed_data_root).expanduser().resolve()
    if not output.is_relative_to(allowed) or not manifest.is_relative_to(allowed):
        raise ValueError("Blind artifacts must remain beneath allowed_data_root")
    if output.exists() or manifest.exists():
        raise FileExistsError("Blind benchmark artifacts are immutable")
    secret = Path(seed_file).read_bytes()
    if len(secret) < 16:
        raise ValueError("Blind seed file must contain at least 16 bytes")
    seed = int.from_bytes(hashlib.sha256(secret).digest(), "big")
    candidates = _load_questions(candidate_path)
    contaminated: list[dict[str, Any]] = []
    for path in contamination_paths:
        contaminated.extend(_load_questions(path))

    rejected = Counter()
    safe: list[dict[str, Any]] = []
    for item in candidates:
        collision = next(
            (
                reason
                for prior in [*contaminated, *safe]
                if (reason := _collision(item, prior)) is not None
            ),
            None,
        )
        if collision:
            rejected[collision] += 1
            continue
        safe.append(item)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in safe:
        grouped[
            (
                str(item.get("category", "")),
                str(item.get("sub_category", item.get("subcategory", ""))),
            )
        ].append(item)
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key in sorted(grouped):
        bucket = sorted(grouped[key], key=lambda row: str(row.get("question_id", "")))
        rng.shuffle(bucket)
        selected.extend(bucket[:questions_per_subcategory])
    spec = {
        "minimum_questions_per_subcategory": questions_per_subcategory,
        "require_all_subcategories": True,
        "require_explicit_validation": True,
        "minimum_difficulty_bands": 3,
        "minimum_answer_types": 2,
        "minimum_template_clusters": 10,
        "minimum_reasoning_structures": 3,
    }
    coverage = validate_evaluation_coverage(selected, spec)
    expected_count = 27 * questions_per_subcategory
    if len(selected) != expected_count:
        raise ValueError(
            f"Blind pool has {len(selected)} safe items; {expected_count} required"
        )
    payload = {
        "schema_version": "1.0",
        "name": "fixed_math_blind_v1",
        "role": "blind_final",
        "questions": selected,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    receipt = {
        "schema_version": "1.0",
        "name": "fixed_math_blind_v1",
        "benchmark_role": "blind_final",
        "question_count": len(selected),
        "subcategory_count": len(grouped),
        "sha256": digest,
        "dedup_policy_version": "benchmark_dedup_v2",
        "dedup_rejections": dict(sorted(rejected.items())),
        "coverage": coverage,
        "allowed_uses": ["frozen_model_final_evaluation"],
        "forbidden_uses": [
            "training",
            "generation",
            "hard_sample_mining",
            "error_directed_generation",
            "prompt_context",
            "hyperparameter_selection",
        ],
    }
    atomic_json(receipt, manifest)
    if os.name != "nt":
        os.chmod(output, 0o600)
        os.chmod(manifest, 0o600)
    return receipt
