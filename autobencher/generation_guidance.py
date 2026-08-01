"""Build and load a leakage-safe 270-record generation guidance set."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .config import DEFAULT_TAXONOMY
from .evaluation_audit import (
    math_ast_signature,
    math_structure_signature,
    normalize_question_text,
    template_signature,
    token_jaccard,
)
from .experiment import atomic_json


VARIATION_AXES = (
    "change the real-world context while preserving the reasoning dependency",
    "change constants and their scale, signs, or divisibility relationships",
    "change the representation between prose, equations, tables, or lists",
    "reverse the unknown and one known quantity to form an inverse problem",
    "add a relevant constraint that still leaves one uniquely checkable answer",
    "change the order in which intermediate quantities must be derived",
    "use a boundary or special case without making the problem trivial",
    "change units or normalization while preserving dimensional consistency",
    "replace a direct computation with an equivalent parameterized form",
    "combine two compatible reasoning steps without copying the source surface form",
)

FORBIDDEN_GUIDANCE_FIELDS = frozenset(
    {
        "question",
        "problem",
        "answer",
        "gold_answer",
        "canonical_answer",
        "display_answer",
        "solution",
        "source_question",
    }
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return _sha256_bytes(value.encode("utf-8"))


def _load_questions(path: str | Path) -> tuple[list[dict[str, Any]], str]:
    source = Path(path).expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    questions = payload.get("questions") if isinstance(payload, Mapping) else None
    if not isinstance(questions, list):
        raise ValueError("Guidance source must contain a questions array")
    return [dict(item) for item in questions], _sha256_bytes(source.read_bytes())


def _candidate_signatures(item: Mapping[str, Any]) -> dict[str, str]:
    question = str(item.get("question", "")).strip()
    if not question:
        raise ValueError("Every guidance source record must contain a question")
    return {
        "normalized": normalize_question_text(question),
        "template": template_signature(question),
        "structure": math_structure_signature(question),
        "ast": math_ast_signature(question),
    }


def _is_too_similar(
    candidate: Mapping[str, Any],
    selected: Iterable[Mapping[str, Any]],
    threshold: float,
) -> bool:
    signatures = candidate["_signatures"]
    for existing in selected:
        other = existing["_signatures"]
        if signatures["normalized"] == other["normalized"]:
            return True
        if signatures["template"] and signatures["template"] == other["template"]:
            return True
        lexical = max(
            token_jaccard(candidate["question"], existing["question"], ngram=1),
            token_jaccard(candidate["question"], existing["question"], ngram=2),
        )
        if lexical >= threshold:
            return True
    return False


def _select_diverse(
    records: list[dict[str, Any]],
    *,
    count: int,
    seed: int,
    similarity_threshold: float,
) -> list[dict[str, Any]]:
    ordered = sorted(
        records,
        key=lambda item: _sha256_text(
            f"{seed}:{item.get('source_dataset')}:{item.get('question_id')}"
        ),
    )
    selected: list[dict[str, Any]] = []
    sources: Counter[str] = Counter()
    answer_types: Counter[str] = Counter()
    difficulty_bands: Counter[str] = Counter()
    reasoning: Counter[str] = Counter()
    while ordered and len(selected) < count:
        eligible = [
            item
            for item in ordered
            if not _is_too_similar(item, selected, similarity_threshold)
        ]
        if not eligible:
            break
        chosen = max(
            eligible,
            key=lambda item: (
                sources[str(item.get("source_dataset", ""))] == 0,
                answer_types[str(item.get("answer_type", ""))] == 0,
                difficulty_bands[str(item.get("difficulty_band", ""))] == 0,
                reasoning[str(item.get("reasoning_structure", ""))] == 0,
                -sources[str(item.get("source_dataset", ""))],
                -answer_types[str(item.get("answer_type", ""))],
                -difficulty_bands[str(item.get("difficulty_band", ""))],
                _sha256_text(f"{seed}:{item.get('question_id')}"),
            ),
        )
        ordered.remove(chosen)
        selected.append(chosen)
        sources[str(chosen.get("source_dataset", ""))] += 1
        answer_types[str(chosen.get("answer_type", ""))] += 1
        difficulty_bands[str(chosen.get("difficulty_band", ""))] += 1
        reasoning[str(chosen.get("reasoning_structure", ""))] += 1
    return selected


def _difficulty_band(item: Mapping[str, Any]) -> str:
    declared = str(item.get("difficulty_band", "")).strip().lower()
    if declared in {"basic", "intermediate", "advanced"}:
        return declared
    difficulty = int(item.get("difficulty", 5))
    if difficulty <= 3:
        return "basic"
    if difficulty <= 6:
        return "intermediate"
    return "advanced"


def build_generation_guidance_payload(
    source_path: str | Path,
    *,
    records_per_subcategory: int = 10,
    seed: int = 2026,
    similarity_threshold: float = 0.72,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Distill source test items into non-verbatim generation blueprints."""
    if records_per_subcategory != len(VARIATION_AXES):
        raise ValueError(
            "Generation guidance v1 requires exactly "
            f"{len(VARIATION_AXES)} records per subcategory"
        )
    if not 0 < similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be in (0, 1]")
    questions, source_sha256 = _load_questions(source_path)
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    rejected = Counter()
    for raw in questions:
        category = str(raw.get("category", ""))
        subcategory = str(raw.get("sub_category", raw.get("subcategory", "")))
        if (
            category not in DEFAULT_TAXONOMY
            or subcategory not in DEFAULT_TAXONOMY[category]
        ):
            rejected["invalid_taxonomy"] += 1
            continue
        try:
            signatures = _candidate_signatures(raw)
        except ValueError:
            rejected["missing_question"] += 1
            continue
        item = dict(raw)
        item["difficulty_band"] = _difficulty_band(item)
        item["_signatures"] = signatures
        buckets[(category, subcategory)].append(item)

    guides = []
    deficits = []
    for category, subcategories in DEFAULT_TAXONOMY.items():
        for subcategory in subcategories:
            selected = _select_diverse(
                buckets[(category, subcategory)],
                count=records_per_subcategory,
                seed=seed,
                similarity_threshold=similarity_threshold,
            )
            if len(selected) < records_per_subcategory:
                deficits.append(
                    f"{category}/{subcategory}={len(selected)}/"
                    f"{records_per_subcategory}"
                )
                continue
            for index, source in enumerate(selected):
                signatures = source["_signatures"]
                reasoning = str(
                    source.get("reasoning_structure", "multi_step_reasoning")
                )
                answer_type = str(source.get("answer_type", "text"))
                band = str(source["difficulty_band"])
                guide_id = f"guide-{len(guides) + 1:03d}"
                guides.append(
                    {
                        "guide_id": guide_id,
                        "category": category,
                        "sub_category": subcategory,
                        "difficulty_band": band,
                        "recommended_answer_type": answer_type,
                        "reasoning_structure": reasoning,
                        "variation_axis": VARIATION_AXES[index],
                        "guidance_instruction": (
                            f"Create a {band} {subcategory} problem using a "
                            f"{reasoning} reasoning pattern and a verifiable "
                            f"{answer_type} answer contract; {VARIATION_AXES[index]}. "
                            "Do not reproduce any public benchmark wording, "
                            "constants, entities, or answer."
                        ),
                        "source_provenance": {
                            "dataset": str(source.get("source_dataset", "")),
                            "split": str(source.get("source_split", "test")),
                            "revision": str(source.get("source_revision", "")),
                            "record_id": str(source.get("source_record_id", "")),
                            "record_sha256": str(
                                source.get("source_record_sha256", "")
                            ),
                        },
                        "source_signatures": {
                            "question_sha256": _sha256_text(
                                str(source.get("question", ""))
                            ),
                            "template_sha256": _sha256_text(
                                signatures["template"]
                            ),
                            "math_structure_sha256": _sha256_text(
                                signatures["structure"]
                            ),
                            "math_ast_sha256": _sha256_text(signatures["ast"]),
                        },
                    }
                )
    if deficits:
        raise ValueError(
            "Generation guidance diversity/coverage is incomplete: "
            + ", ".join(deficits[:12])
        )
    expected = records_per_subcategory * sum(
        len(subcategories) for subcategories in DEFAULT_TAXONOMY.values()
    )
    if len(guides) != expected:
        raise AssertionError(f"Expected {expected} guidance records, got {len(guides)}")
    payload = {
        "schema_version": "1.0",
        "name": "open_source_generation_guidance_v1",
        "version": "1.0",
        "role": "generation_guidance",
        "permissions": {
            "generator_prompt": True,
            "training": False,
            "evaluation": False,
            "hard_pool": False,
        },
        "source_text_included": False,
        "source_answers_included": False,
        "diversity_policy": {
            "reject_exact_normalized_match": True,
            "reject_template_signature_match": True,
            "lexical_jaccard_threshold": similarity_threshold,
            "variation_axis_count": len(VARIATION_AXES),
        },
        "records": guides,
    }
    manifest = {
        "schema_version": "1.0",
        "name": payload["name"],
        "version": payload["version"],
        "question_count": len(guides),
        "subcategory_count": len(DEFAULT_TAXONOMY_FLAT),
        "records_per_subcategory": records_per_subcategory,
        "selection_seed": seed,
        "source_candidate_sha256": source_sha256,
        "similarity_threshold": similarity_threshold,
        "source_text_included": False,
        "source_answers_included": False,
        "rejected_source_records": dict(sorted(rejected.items())),
    }
    return payload, manifest


DEFAULT_TAXONOMY_FLAT = tuple(
    (category, subcategory)
    for category, subcategories in DEFAULT_TAXONOMY.items()
    for subcategory in subcategories
)


def _assert_no_leaking_fields(value: Any) -> None:
    if isinstance(value, Mapping):
        leaking = FORBIDDEN_GUIDANCE_FIELDS & set(value)
        if leaking:
            raise ValueError(
                "Generation guidance contains forbidden source fields: "
                + ", ".join(sorted(leaking))
            )
        for nested in value.values():
            _assert_no_leaking_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_leaking_fields(nested)


def validate_generation_guidance(
    payload: Mapping[str, Any],
    *,
    expected_count: int = 270,
    records_per_subcategory: int = 10,
) -> list[dict[str, Any]]:
    if payload.get("role") != "generation_guidance":
        raise ValueError("Generation guidance role must be generation_guidance")
    if payload.get("source_text_included") is not False:
        raise ValueError("Generation guidance must not include source question text")
    if payload.get("source_answers_included") is not False:
        raise ValueError("Generation guidance must not include source answers")
    if payload.get("permissions") != {
        "generator_prompt": True,
        "training": False,
        "evaluation": False,
        "hard_pool": False,
    }:
        raise ValueError("Generation guidance permissions are unsafe")
    records = payload.get("records")
    if not isinstance(records, list) or len(records) != expected_count:
        raise ValueError(
            f"Generation guidance must contain exactly {expected_count} records"
        )
    _assert_no_leaking_fields(records)
    identifiers = set()
    counts = Counter()
    for raw in records:
        if not isinstance(raw, Mapping):
            raise ValueError("Every generation guidance record must be an object")
        record = dict(raw)
        guide_id = str(record.get("guide_id", "")).strip()
        category = str(record.get("category", ""))
        subcategory = str(record.get("sub_category", ""))
        if not re.fullmatch(r"guide-[0-9]{3}", guide_id) or guide_id in identifiers:
            raise ValueError(f"Invalid or duplicate guide_id: {guide_id!r}")
        if (
            category not in DEFAULT_TAXONOMY
            or subcategory not in DEFAULT_TAXONOMY[category]
        ):
            raise ValueError(f"Invalid guidance taxonomy: {category}/{subcategory}")
        instruction = str(record.get("guidance_instruction", "")).strip()
        provenance = record.get("source_provenance")
        signatures = record.get("source_signatures")
        if not instruction or not isinstance(provenance, Mapping):
            raise ValueError(f"Guidance record is incomplete: {guide_id}")
        if not str(provenance.get("dataset", "")).strip():
            raise ValueError(f"Guidance source dataset is missing: {guide_id}")
        if str(provenance.get("split", "")).strip().lower() != "test":
            raise ValueError(f"Guidance source must use a test split: {guide_id}")
        revision = str(provenance.get("revision", "")).strip().lower()
        if len(revision) != 40 or set(revision) - set("0123456789abcdef"):
            raise ValueError(f"Guidance source revision is invalid: {guide_id}")
        if not str(provenance.get("record_id", "")).strip():
            raise ValueError(f"Guidance source record ID is missing: {guide_id}")
        record_sha256 = str(provenance.get("record_sha256", "")).strip().lower()
        if len(record_sha256) != 64 or set(record_sha256) - set(
            "0123456789abcdef"
        ):
            raise ValueError(f"Guidance source record hash is invalid: {guide_id}")
        if not isinstance(signatures, Mapping):
            raise ValueError(f"Guidance signatures are missing: {guide_id}")
        for field in (
            "question_sha256",
            "template_sha256",
            "math_structure_sha256",
            "math_ast_sha256",
        ):
            value = str(signatures.get(field, ""))
            if len(value) != 64 or set(value) - set("0123456789abcdef"):
                raise ValueError(f"Invalid {field} for {guide_id}")
        identifiers.add(guide_id)
        counts[(category, subcategory)] += 1
    invalid_counts = {
        pair: counts[pair]
        for pair in DEFAULT_TAXONOMY_FLAT
        if counts[pair] != records_per_subcategory
    }
    if invalid_counts:
        raise ValueError(f"Generation guidance coverage is invalid: {invalid_counts}")
    return [dict(record) for record in records]


def write_generation_guidance(
    source_path: str | Path,
    output_path: str | Path,
    *,
    allowed_data_root: str | Path,
    records_per_subcategory: int = 10,
    seed: int = 2026,
    similarity_threshold: float = 0.72,
) -> dict[str, Any]:
    root = Path(allowed_data_root).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    output = Path(output_path).expanduser().resolve()
    if not source.is_relative_to(root) or not output.is_relative_to(root):
        raise ValueError(
            "Guidance source and output must remain under allowed_data_root"
        )
    if output.exists():
        raise FileExistsError(f"Immutable guidance dataset already exists: {output}")
    source_manifest_path = source.with_suffix(source.suffix + ".manifest.json")
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            "Open-source candidate manifest does not exist: "
            f"{source_manifest_path}"
        )
    source_manifest = json.loads(
        source_manifest_path.read_text(encoding="utf-8")
    )
    source_sha256 = _sha256_bytes(source.read_bytes())
    if str(source_manifest.get("sha256", "")).lower() != source_sha256:
        raise ValueError("Open-source candidate hash does not match its manifest")
    payload, manifest = build_generation_guidance_payload(
        source,
        records_per_subcategory=records_per_subcategory,
        seed=seed,
        similarity_threshold=similarity_threshold,
    )
    validate_generation_guidance(
        payload,
        expected_count=len(DEFAULT_TAXONOMY_FLAT) * records_per_subcategory,
        records_per_subcategory=records_per_subcategory,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(payload, output)
    manifest.update(
        {
            "path": output.as_posix(),
            "sha256": _sha256_bytes(output.read_bytes()),
            "source_candidate_manifest_sha256": _sha256_bytes(
                source_manifest_path.read_bytes()
            ),
        }
    )
    atomic_json(manifest, output.with_suffix(output.suffix + ".manifest.json"))
    return manifest


def load_generation_guidance_context(
    config: Mapping[str, Any],
    *,
    category: str,
    subcategory: str,
    target_difficulty: int,
    selection_key: str,
) -> tuple[str, list[str]]:
    settings = config.get("generation_guidance", {})
    if not bool(settings.get("enabled", False)):
        return "", []
    path = Path(str(settings["dataset_path"])).expanduser().resolve()
    if bool(settings.get("require_manifest", True)):
        manifest_path = path.with_suffix(path.suffix + ".manifest.json")
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"Generation guidance manifest does not exist: {manifest_path}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        observed_sha256 = _sha256_bytes(path.read_bytes())
        if str(manifest.get("sha256", "")).lower() != observed_sha256:
            raise ValueError("Generation guidance hash does not match its manifest")
        if int(manifest.get("question_count", 0)) != int(
            settings["expected_record_count"]
        ):
            raise ValueError("Generation guidance manifest count is inconsistent")
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = validate_generation_guidance(
        payload,
        expected_count=int(settings["expected_record_count"]),
        records_per_subcategory=int(settings["records_per_subcategory"]),
    )
    matches = [
        record
        for record in records
        if record["category"] == category
        and record["sub_category"] == subcategory
    ]
    maximum = int(settings["max_prompt_records"])
    ordered = sorted(
        matches,
        key=lambda record: (
            abs(
                {"basic": 3, "intermediate": 5, "advanced": 7}.get(
                    record["difficulty_band"],
                    5,
                )
                - int(target_difficulty)
            ),
            _sha256_text(f"{selection_key}:{record['guide_id']}"),
        ),
    )[:maximum]
    if not ordered:
        raise ValueError(f"No generation guidance for {category}/{subcategory}")
    lines = [
        "Static open-source-derived guidance. The source question and answer "
        "are intentionally withheld:"
    ]
    for record in ordered:
        lines.append(
            f"- {record['guide_id']}: {record['guidance_instruction']}"
        )
    return "\n".join(lines), [record["guide_id"] for record in ordered]
