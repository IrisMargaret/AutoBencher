"""Mixed training-dataset construction, deduplication, and noise reporting."""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Mapping

from .coverage import largest_remainder
from .similarity import (
    SimilarityBatch,
    build_similarity_batch,
)


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


def _pair_similarity_details(
    left: Any,
    right: Any,
    batch: SimilarityBatch,
    left_index: int,
    right_index: int,
) -> dict[str, Any]:
    return {
        "token_unigram_similarity": token_jaccard(
            left,
            right,
            ngram=1,
        ),
        "token_bigram_similarity": token_jaccard(
            left,
            right,
            ngram=2,
        ),
        "tfidf_similarity": tfidf_cosine(left, right),
        **batch.pair(left_index, right_index),
    }


def _holdout_leakage_results(
    records: list[Mapping[str, Any]],
    holdout_records: list[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> list[tuple[str | None, dict[str, Any]]]:
    """Batch holdout comparison so semantic embeddings are computed once."""
    if not holdout_records:
        return [(None, {}) for _ in records]
    questions = [record.get("question", "") for record in records]
    holdout_questions = [
        record.get("question", "") for record in holdout_records
    ]
    results: list[tuple[str | None, dict[str, Any]]] = []
    unresolved = []
    for record_index, question in enumerate(questions):
        normalized = normalize_question_text(question)
        template = template_signature(question)
        result = None
        for holdout_index, holdout in enumerate(holdout_records):
            holdout_question = holdout_questions[holdout_index]
            holdout_id = holdout.get("question_id", holdout.get("id"))
            if normalized == normalize_question_text(holdout_question):
                result = (
                    "holdout_exact_match",
                    {
                        "holdout_question_id": holdout_id,
                        "similarity": 1.0,
                    },
                )
                break
            if (
                config["dataset"]["holdout_exact_template_rejection"]
                and template
                and template == template_signature(holdout_question)
            ):
                result = (
                    "holdout_template_match",
                    {
                        "holdout_question_id": holdout_id,
                        "similarity": 1.0,
                    },
                )
                break
        results.append(result or (None, {}))
        if result is None:
            unresolved.append(record_index)
    if not unresolved:
        return results

    batch = build_similarity_batch(
        [*questions, *holdout_questions],
        config["dataset"],
    )
    near_threshold = float(
        config["dataset"]["holdout_near_duplicate_threshold"]
    )
    semantic_threshold = float(
        config["dataset"]["holdout_semantic_similarity_threshold"]
    )
    minhash_threshold = float(
        config["dataset"]["holdout_text_dedup_similarity_threshold"]
    )
    holdout_offset = len(questions)
    for record_index in unresolved:
        question = questions[record_index]
        for holdout_index, holdout in enumerate(holdout_records):
            details = _pair_similarity_details(
                question,
                holdout_questions[holdout_index],
                batch,
                record_index,
                holdout_offset + holdout_index,
            )
            lexical_similarity = max(
                details["token_unigram_similarity"],
                details["token_bigram_similarity"],
                details["tfidf_similarity"],
            )
            minhash_hit = bool(
                config["dataset"]["text_dedup_enabled"]
            ) and details["minhash_similarity"] >= minhash_threshold
            semantic_score = details[
                "sentence_transformers_similarity"
            ]
            semantic_hit = (
                semantic_score is not None
                and semantic_score >= semantic_threshold
            )
            if (
                lexical_similarity >= near_threshold
                or minhash_hit
                or semantic_hit
            ):
                results[record_index] = (
                    (
                        "holdout_semantic_duplicate"
                        if semantic_hit
                        else "holdout_near_duplicate"
                    ),
                    {
                        "holdout_question_id": holdout.get(
                            "question_id",
                            holdout.get("id"),
                        ),
                        **details,
                    },
                )
                break
    return results


def holdout_leakage_reason(
    record: Mapping[str, Any],
    holdout_records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> tuple[str | None, dict[str, Any]]:
    """Return a fail-closed reason when a training question overlaps holdout."""
    return _holdout_leakage_results(
        [record],
        [dict(item) for item in holdout_records],
        config,
    )[0]


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
    if dataset_config["require_gold_reasoning_steps"]:
        _, reasoning_error = _validated_gold_reasoning_steps(
            record,
            config,
        )
        if reasoning_error:
            reasons.append(reasoning_error)
    return reasons


def _validated_gold_reasoning_steps(
    record: Mapping[str, Any],
    config: Mapping[str, Any],
) -> tuple[list[str], str | None]:
    dataset_config = config["dataset"]
    raw_steps = record.get("gold_reasoning_summary")
    if not isinstance(raw_steps, list):
        return [], "missing_gold_reasoning_steps"
    steps = [
        re.sub(r"\s+", " ", str(step or "")).strip()
        for step in raw_steps
    ]
    minimum = int(dataset_config["min_gold_reasoning_steps"])
    maximum = int(dataset_config["max_gold_reasoning_steps"])
    maximum_chars = int(
        dataset_config["max_gold_reasoning_chars_per_step"]
    )
    if not minimum <= len(steps) <= maximum or any(not step for step in steps):
        return [], "invalid_gold_reasoning_steps"
    forbidden = (
        "```",
        "<|",
        "|>",
        "system:",
        "assistant:",
        "user:",
        "human:",
        "question_json",
        "python_code",
        "ignore previous",
        "browse",
        "search the web",
        "tool call",
    )
    if any(
        len(step) > maximum_chars
        or any(marker in step.lower() for marker in forbidden)
        for step in steps
    ):
        return [], "unsafe_gold_reasoning_steps"
    return steps, None


def build_training_dataset(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    seed: int | None = None,
    holdout_records: Iterable[Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    source_records = [dict(record) for record in records]
    holdout_records = [
        dict(record) for record in (holdout_records or [])
    ]
    rejection_counts: Counter[str] = Counter()
    rejected = []
    clean = []
    leakage_results = _holdout_leakage_results(
        source_records,
        holdout_records,
        config,
    )
    for record, (leakage_reason, leakage_details) in zip(
        source_records,
        leakage_results,
    ):
        reasons = _rejection_reasons(record, config)
        if leakage_reason:
            reasons.append(leakage_reason)
        if reasons:
            rejection_counts.update(reasons)
            rejection = {"record": record, "reasons": reasons}
            if leakage_details:
                rejection["holdout_leakage"] = leakage_details
            rejected.append(rejection)
        else:
            reasoning_steps, _ = _validated_gold_reasoning_steps(
                record,
                config,
            )
            record["gold_reasoning_summary"] = reasoning_steps
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
    similarity_batch = build_similarity_batch(
        [record["question"] for record in exact_unique],
        config["dataset"],
    )
    minhash_enabled = bool(config["dataset"]["text_dedup_enabled"])
    minhash_threshold = float(
        config["dataset"]["text_dedup_similarity_threshold"]
    )
    embedding_threshold = float(
        config["dataset"]["semantic_similarity_threshold"]
    )
    near_unique = []
    near_unique_indices = []
    for record_index, record in enumerate(exact_unique):
        lsh_candidates = set(
            similarity_batch.minhash_candidates(record_index)
        )
        comparison_candidates = sorted(
            zip(near_unique, near_unique_indices),
            key=lambda item: (
                item[1] not in lsh_candidates,
                item[1],
            ),
        )
        duplicate_match = next(
            (
                (existing, existing_index)
                for existing, existing_index in comparison_candidates
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
                    or (
                        minhash_enabled
                        and similarity_batch.pair(
                            record_index,
                            existing_index,
                        )["minhash_similarity"]
                        >= minhash_threshold
                    )
                    or (
                        semantic_enabled
                        and similarity_batch.pair(
                            record_index,
                            existing_index,
                        )["sentence_transformers_similarity"]
                        is not None
                        and similarity_batch.pair(
                            record_index,
                            existing_index,
                        )["sentence_transformers_similarity"]
                        >= embedding_threshold
                    )
                )
            ),
            None,
        )
        if duplicate_match is not None:
            duplicate, duplicate_index = duplicate_match
            similarity_details = _pair_similarity_details(
                record["question"],
                duplicate["question"],
                similarity_batch,
                record_index,
                duplicate_index,
            )
            rejection_counts["near_duplicate"] += 1
            rejected.append(
                {
                    "record": record,
                    "reasons": ["near_duplicate"],
                    "duplicate_similarity": similarity_details,
                }
            )
        else:
            near_unique.append(record)
            near_unique_indices.append(record_index)

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
    rng = random.Random(
        int(seed if seed is not None else config["experiment"]["seed"])
    )
    strict_ratio = bool(mix["strict_correct_incorrect_ratio"])
    selected: list[dict[str, Any]] = []
    selected_counts: Counter[str] = Counter()
    if strict_ratio:
        correct_fraction = Fraction(
            str(float(mix["correct_retention_samples"]))
        ).limit_denominator(100)
        block_size = correct_fraction.denominator
        correct_per_block = correct_fraction.numerator
        wrong_per_block = block_size - correct_per_block
        correct_pool = sorted(
            (
                record
                for record in deduplicated
                if bool(record.get("is_correct"))
            ),
            key=lambda item: (-item["quality_score"], item["exact_signature"]),
        )
        wrong_pool = sorted(
            (
                record
                for record in deduplicated
                if not bool(record.get("is_correct"))
            ),
            key=lambda item: (-item["quality_score"], item["exact_signature"]),
        )
        block_count = min(
            len(correct_pool) // correct_per_block,
            len(wrong_pool) // wrong_per_block,
            len(deduplicated) // block_size,
        )
        target_total = block_count * block_size
        correct_target = block_count * correct_per_block
        wrong_target = block_count * wrong_per_block
        requested = largest_remainder(
            wrong_target,
            {
                "incorrect_boundary_samples": float(
                    mix["incorrect_boundary_samples"]
                ),
                "coverage_repair_samples": float(
                    mix["coverage_repair_samples"]
                ),
                "format_instruction_samples": float(
                    mix["format_instruction_samples"]
                ),
            },
        )
        requested["correct_retention_samples"] = correct_target
        chosen_correct = correct_pool[:correct_target]
        selected.extend(chosen_correct)
        selected_counts.update(
            record["training_source"] for record in chosen_correct
        )
        wrong_by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for record in wrong_pool:
            wrong_by_source[record["training_source"]].append(record)
        unused_wrong = []
        for source in (
            "incorrect_boundary_samples",
            "coverage_repair_samples",
            "format_instruction_samples",
        ):
            candidates = wrong_by_source.get(source, [])
            chosen = candidates[: requested[source]]
            selected.extend(chosen)
            selected_counts.update(
                record["training_source"] for record in chosen
            )
            unused_wrong.extend(candidates[requested[source]:])
        missing_wrong = wrong_target - sum(
            not bool(record.get("is_correct")) for record in selected
        )
        if missing_wrong:
            rng.shuffle(unused_wrong)
            fallback = unused_wrong[:missing_wrong]
            selected.extend(fallback)
            selected_counts.update(
                record["training_source"] for record in fallback
            )
    else:
        target_total = len(deduplicated)
        requested = largest_remainder(
            target_total,
            {
                "incorrect_boundary_samples": float(
                    mix["incorrect_boundary_samples"]
                ),
                "correct_retention_samples": float(
                    mix["correct_retention_samples"]
                ),
                "coverage_repair_samples": float(
                    mix["coverage_repair_samples"]
                ),
                "format_instruction_samples": float(
                    mix["format_instruction_samples"]
                ),
            },
        )
        fallback_pool = []
        for source, target in requested.items():
            candidates = sorted(
                grouped.get(source, []),
                key=lambda item: (
                    -item["quality_score"],
                    item["exact_signature"],
                ),
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
                record["training_source"]
                for record in fallback_pool[:remaining]
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
                        [],
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
        "similarity_backend": {
            "minhash": (
                similarity_batch.minhash_backend
                if minhash_enabled
                else "disabled"
            ),
            "minhash_error": similarity_batch.minhash_error,
            "minhash_lsh": bool(similarity_batch.minhash_lsh),
            "sentence_transformers": (
                similarity_batch.semantic_backend
            ),
            "sentence_transformers_error": (
                similarity_batch.semantic_error
            ),
        },
        "requested_mix_counts": requested,
        "selected_mix_counts": dict(selected_counts),
        "selected_correct_count": sum(
            bool(record.get("is_correct")) for record in selected
        ),
        "selected_incorrect_count": sum(
            not bool(record.get("is_correct")) for record in selected
        ),
        "selected_correct_fraction": (
            sum(bool(record.get("is_correct")) for record in selected)
            / len(selected)
            if selected
            else 0.0
        ),
        "strict_correct_incorrect_ratio": strict_ratio,
        "strict_ratio_satisfied": (
            not selected
            or abs(
                sum(bool(record.get("is_correct")) for record in selected)
                / len(selected)
                - float(mix["correct_retention_samples"])
            )
            < 1e-12
        ),
        "minimum_sample_requirement_met": (
            len(selected) >= int(mix["minimum_samples"])
        ),
        "holdout_question_count": len(holdout_records),
        "holdout_leakage_rejected_count": sum(
            any(reason.startswith("holdout_") for reason in item["reasons"])
            for item in rejected
        ),
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
