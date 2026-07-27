"""Mixed training-dataset construction, deduplication, and noise reporting."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .coverage import largest_remainder


def normalize_question_text(text: Any) -> str:
    normalized = re.sub(r"\s+", " ", str(text or "").strip().lower())
    return re.sub(r"[^\w\s%+\-*/=().<>{}\[\],]", "", normalized)


def template_signature(text: Any) -> str:
    normalized = normalize_question_text(text)
    normalized = re.sub(r"\b\d+(?:\.\d+)?(?:/\d+)?\b", "<NUM>", normalized)
    normalized = re.sub(
        r"\b(?:km|cm|mm|m|kg|g|hours?|minutes?|seconds?|degrees?)\b",
        "<UNIT>",
        normalized,
    )
    normalized = re.sub(r"\b[a-z]\b", "<VAR>", normalized)
    return normalized


def exact_signature(record: Mapping[str, Any]) -> str:
    payload = "|".join(
        (
            normalize_question_text(record.get("question", "")),
            normalize_question_text(
                record.get("gold_answer", record.get("canonical_answer", ""))
            ),
            str(record.get("category", "")),
            str(record.get("sub_category", record.get("subcategory", ""))),
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def token_jaccard(left: Any, right: Any, ngram: int = 2) -> float:
    def shingles(value: Any) -> set[tuple[str, ...]]:
        tokens = normalize_question_text(value).split()
        if len(tokens) < ngram:
            return {tuple(tokens)} if tokens else set()
        return {
            tuple(tokens[index:index + ngram])
            for index in range(len(tokens) - ngram + 1)
        }

    left_set, right_set = shingles(left), shingles(right)
    if not left_set and not right_set:
        return 1.0
    union = left_set | right_set
    return len(left_set & right_set) / len(union) if union else 0.0


def tfidf_cosine(left: Any, right: Any) -> float:
    """Compute a dependency-free two-document TF-IDF cosine similarity."""
    documents = [
        normalize_question_text(left).split(),
        normalize_question_text(right).split(),
    ]
    if not documents[0] and not documents[1]:
        return 1.0
    vocabulary = set(documents[0]) | set(documents[1])
    vectors = []
    for document in documents:
        counts = Counter(document)
        length = max(1, len(document))
        vector = {}
        for token in vocabulary:
            document_frequency = sum(token in candidate for candidate in documents)
            inverse_document_frequency = math.log(
                (1 + len(documents)) / (1 + document_frequency)
            ) + 1
            vector[token] = counts[token] / length * inverse_document_frequency
        vectors.append(vector)
    numerator = sum(vectors[0][token] * vectors[1][token] for token in vocabulary)
    norms = [
        math.sqrt(sum(value * value for value in vector.values()))
        for vector in vectors
    ]
    if not all(norms):
        return 0.0
    return numerator / (norms[0] * norms[1])


def _quality_score(record: Mapping[str, Any]) -> float:
    confidence = float(
        record.get(
            "attribution_confidence",
            record.get("evaluator_confidence", 1.0),
        )
    )
    subcategory_accuracy = float(record.get("sub_category_accuracy", 0.2))
    boundary = math.exp(-abs(subcategory_accuracy - 0.2) / 0.15)
    return confidence * 0.6 + boundary * 0.3 + min(
        len(str(record.get("question", ""))) / 300,
        0.1,
    )


def _candidate_source(record: Mapping[str, Any]) -> str:
    if record.get("format_only_error") or record.get("parse_status") not in {
        None,
        "success",
    }:
        return "format_instruction_samples"
    if bool(record.get("is_correct")):
        return "correct_retention_samples"
    if record.get("sample_grade") == "train_eligible" or (
        0.1 <= float(record.get("sub_category_accuracy", 0.2)) <= 0.4
    ):
        return "incorrect_boundary_samples"
    return "coverage_repair_samples"


def _rejection_reasons(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> list[str]:
    reasons = []
    dataset_config = config["dataset"]
    if record.get("question_parse_success", True) is not True:
        reasons.append("question_parse_failed")
    if (
        dataset_config["filter_invalid_answers"]
        and record.get("answer_validation_success", True) is not True
    ):
        reasons.append("answer_validation_failed")
    confidence = float(
        record.get(
            "evaluator_confidence",
            record.get("attribution_confidence", 1.0),
        )
    )
    if confidence < float(dataset_config["evaluator_confidence_threshold"]):
        reasons.append("low_evaluator_confidence")
    if dataset_config["filter_ambiguous_samples"] and record.get("ambiguous"):
        reasons.append("ambiguous_question")
    if dataset_config["filter_tool_violations"] and record.get("tool_violation"):
        reasons.append("tool_violation")
    if record.get("contains_irrelevant_content"):
        reasons.append("irrelevant_output")
    if record.get("contains_prompt_echo"):
        reasons.append("prompt_echo")
    if not str(record.get("question", "")).strip():
        reasons.append("empty_question")
    if not str(
        record.get("gold_answer", record.get("canonical_answer", ""))
    ).strip():
        reasons.append("empty_answer")
    return reasons


def build_training_dataset(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source_records = [dict(record) for record in records]
    rejection_counts: Counter[str] = Counter()
    rejected = []
    clean = []
    for record in source_records:
        reasons = _rejection_reasons(record, config)
        if reasons:
            rejection_counts.update(reasons)
            rejected.append({"record": record, "reasons": reasons})
        else:
            record["exact_signature"] = exact_signature(record)
            record["template_signature"] = template_signature(record.get("question"))
            record["training_source"] = _candidate_source(record)
            record["quality_score"] = _quality_score(record)
            clean.append(record)

    exact_seen = set()
    exact_unique = []
    for record in sorted(clean, key=lambda item: -item["quality_score"]):
        if (
            config["dataset"]["exact_dedup"]
            and record["exact_signature"] in exact_seen
        ):
            rejection_counts["exact_duplicate"] += 1
            rejected.append({"record": record, "reasons": ["exact_duplicate"]})
            continue
        exact_seen.add(record["exact_signature"])
        exact_unique.append(record)

    text_near_enabled = bool(config["dataset"].get("text_near_dedup", False))
    threshold = float(config["dataset"]["near_duplicate_threshold"])
    semantic_enabled = bool(config["dataset"].get("semantic_dedup", False))
    semantic_threshold = float(
        config["dataset"].get("semantic_similarity_threshold", 1.0)
    )
    near_unique = []
    for record in exact_unique:
        duplicate = next(
            (
                existing
                for existing in near_unique
                if (
                    (
                        text_near_enabled
                        and token_jaccard(
                            record["question"],
                            existing["question"],
                        )
                        >= threshold
                    )
                    or (
                        semantic_enabled
                        and tfidf_cosine(
                            record["question"],
                            existing["question"],
                        )
                        >= semantic_threshold
                    )
                )
            ),
            None,
        )
        if duplicate is not None:
            rejection_counts["near_duplicate"] += 1
            rejected.append({"record": record, "reasons": ["near_duplicate"]})
        else:
            near_unique.append(record)

    cluster_limit = int(config["dataset"]["max_samples_per_template_cluster"])
    template_counts: Counter[str] = Counter()
    deduplicated = []
    for record in near_unique:
        signature = record["template_signature"]
        if (
            config["dataset"]["template_dedup"]
            and template_counts[signature] >= cluster_limit
        ):
            rejection_counts["template_duplicate"] += 1
            rejected.append({"record": record, "reasons": ["template_duplicate"]})
            continue
        template_counts[signature] += 1
        deduplicated.append(record)

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in deduplicated:
        grouped[record["training_source"]].append(record)
    mix = config["training_mix"]
    target_total = len(deduplicated)
    requested = largest_remainder(
        target_total,
        {
            "incorrect_boundary_samples": float(
                mix["incorrect_boundary_samples"]
            ),
            "correct_retention_samples": float(mix["correct_retention_samples"]),
            "coverage_repair_samples": float(mix["coverage_repair_samples"]),
            "format_instruction_samples": float(mix["format_instruction_samples"]),
        },
    )
    rng = random.Random(
        int(seed if seed is not None else config["experiment"]["seed"])
    )
    selected = []
    fallback_pool = []
    selected_counts: Counter[str] = Counter()
    for source, target in requested.items():
        candidates = sorted(
            grouped.get(source, []),
            key=lambda item: (-item["quality_score"], item["exact_signature"]),
        )
        chosen = candidates[:target]
        selected.extend(chosen)
        selected_counts[source] += len(chosen)
        fallback_pool.extend(candidates[target:])
    remaining = target_total - len(selected)
    if remaining:
        rng.shuffle(fallback_pool)
        selected.extend(fallback_pool[:remaining])
        selected_counts.update(
            record["training_source"] for record in fallback_pool[:remaining]
        )
    selected = sorted(
        selected,
        key=lambda item: (item["training_source"], item["exact_signature"]),
    )
    alpaca = [
        {
            "instruction": (
                "Solve the math problem and return a JSON object with "
                "reasoning_summary, final_answer, answer_type, and confidence."
            ),
            "input": str(record["question"]),
            "output": json.dumps(
                {
                    "reasoning_summary": record.get(
                        "gold_reasoning_summary",
                        ["Apply the relevant mathematical method and verify the result."],
                    ),
                    "final_answer": str(
                        record.get(
                            "gold_answer",
                            record.get("canonical_answer", ""),
                        )
                    ),
                    "answer_type": str(record.get("answer_type", "text")),
                    "confidence": 1.0,
                },
                ensure_ascii=False,
            ),
            "_metadata": {
                "training_source": record["training_source"],
                "exact_signature": record["exact_signature"],
                "template_signature": record["template_signature"],
                "category": record.get("category"),
                "sub_category": record.get(
                    "sub_category",
                    record.get("subcategory"),
                ),
            },
        }
        for record in selected
    ]
    manifest = {
        "raw_candidate_count": len(source_records),
        "noise_clean_count": len(clean),
        "deduplicated_count": len(deduplicated),
        "selected_count": len(alpaca),
        "rejected_count": len(rejected),
        "requested_mix_counts": requested,
        "selected_mix_counts": dict(selected_counts),
        "rejection_reasons": dict(sorted(rejection_counts.items())),
        "exact_unique_count": len(exact_unique),
        "near_unique_count": len(near_unique),
        "template_cluster_count": len(template_counts),
        "near_duplicate_method": {
            "token_jaccard_threshold": threshold,
            "token_jaccard_enabled": text_near_enabled,
            "tfidf_cosine_enabled": semantic_enabled,
            "tfidf_cosine_threshold": semantic_threshold,
        },
    }
    return alpaca, manifest, rejected


def write_alpaca_jsonl(
    records: Iterable[Mapping[str, Any]],
    path: str | Path,
) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            output_record = {
                key: record[key]
                for key in ("instruction", "input", "output")
            }
            handle.write(json.dumps(output_record, ensure_ascii=False))
            handle.write("\n")
            count += 1
        handle.flush()
        import os

        os.fsync(handle.fileno())
    temporary.replace(path)
    return count
