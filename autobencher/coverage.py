"""Coverage metrics, deterministic quotas, and adaptive sampling."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from typing import Any, Iterable, Mapping


def taxonomy_items(config: Mapping[str, Any]) -> list[tuple[str, str, dict[str, Any]]]:
    default_quota = int(config["coverage"]["default_min_quota"])
    items = []
    for category, subcategories in config["taxonomy"].items():
        for subcategory, metadata in subcategories.items():
            item = dict(metadata or {})
            item.setdefault("min_quota", default_quota)
            item.setdefault("base_weight", 1.0)
            items.append((str(category), str(subcategory), item))
    return items


def largest_remainder(total: int, proportions: Mapping[str, float]) -> dict[str, int]:
    if total < 0:
        raise ValueError("total must be non-negative")
    if not proportions:
        return {}
    weight_sum = sum(float(value) for value in proportions.values())
    if weight_sum <= 0:
        raise ValueError("proportions must contain a positive weight")
    exact = {
        key: total * float(value) / weight_sum
        for key, value in proportions.items()
    }
    allocation = {key: int(math.floor(value)) for key, value in exact.items()}
    remaining = total - sum(allocation.values())
    order = sorted(
        exact,
        key=lambda key: (-(exact[key] - allocation[key]), str(key)),
    )
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


def _record_subcategory(record: Mapping[str, Any]) -> str:
    return str(
        record.get(
            "sub_category",
            record.get("subcategory", record.get("target_subcategory", "")),
        )
    )


def coverage_metrics(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    items = taxonomy_items(config)
    known = [subcategory for _, subcategory, _ in items]
    counts = Counter(
        _record_subcategory(record)
        for record in records
        if _record_subcategory(record) in known
    )
    quotas = {subcategory: int(metadata["min_quota"]) for _, subcategory, metadata in items}
    total_types = len(known)
    raw_count = sum(counts[name] > 0 for name in known)
    effective_count = sum(counts[name] >= quotas[name] for name in known)
    total_samples = sum(counts.values())
    probabilities = [
        counts[name] / total_samples if total_samples else 0.0
        for name in known
    ]
    if total_types <= 1 or total_samples == 0:
        normalized_entropy = 0.0 if total_samples == 0 else 1.0
    else:
        entropy = -sum(p * math.log(p) for p in probabilities if p > 0)
        normalized_entropy = entropy / math.log(total_types)
    uniform = 1.0 / total_types if total_types else 0.0
    midpoint = [(p + uniform) / 2 for p in probabilities]
    kl_p = sum(
        p * math.log(p / m)
        for p, m in zip(probabilities, midpoint)
        if p > 0 and m > 0
    )
    kl_u = sum(
        uniform * math.log(uniform / m)
        for m in midpoint
        if uniform > 0 and m > 0
    )
    js_divergence = 0.5 * (kl_p + kl_u)
    count_values = [counts[name] for name in known]
    mean_count = total_samples / total_types if total_types else 0.0
    variance = (
        sum((value - mean_count) ** 2 for value in count_values) / total_types
        if total_types
        else 0.0
    )
    cv = math.sqrt(variance) / mean_count if mean_count else 0.0
    nonzero = [value for value in count_values if value > 0]
    max_min_ratio = (
        max(nonzero) / min(nonzero)
        if nonzero
        else 0.0
    )
    raw_coverage = raw_count / total_types if total_types else 0.0
    effective_coverage = effective_count / total_types if total_types else 0.0
    return {
        "subcategory_coverage": raw_coverage,
        "raw_subcategory_coverage": raw_coverage,
        "effective_subcategory_coverage": effective_coverage,
        "covered_subcategory_count": raw_count,
        "quota_satisfied_subcategory_count": effective_count,
        "total_subcategory_count": total_types,
        "normalized_entropy": normalized_entropy,
        "js_divergence_to_uniform": js_divergence,
        "subcategory_count_cv": cv,
        "max_min_count_ratio": max_min_ratio,
        "subcategory_counts": {name: counts[name] for name in known},
        "subcategory_quotas": quotas,
    }


def beta_binomial_state(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[tuple[str, int], dict[str, float]]:
    adaptive = config["adaptive_sampling"]
    prior_alpha = float(adaptive["beta_prior_alpha"])
    prior_beta = float(adaptive["beta_prior_beta"])
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        key = (_record_subcategory(record), int(record.get("difficulty", 5)))
        grouped[key].append(record)
    state = {}
    for key, group in grouped.items():
        correct = sum(bool(item.get("is_correct")) for item in group)
        incorrect = len(group) - correct
        alpha = prior_alpha + correct
        beta = prior_beta + incorrect
        denominator = alpha + beta
        state[key] = {
            "correct_count": correct,
            "incorrect_count": incorrect,
            "alpha": alpha,
            "beta": beta,
            "posterior_mean": alpha / denominator,
            "posterior_variance": (
                alpha * beta / (denominator ** 2 * (denominator + 1))
            ),
            "observation_count": len(group),
        }
    return state


def adaptive_priority(
    subcategory: str,
    difficulty: int,
    current_count: int,
    min_quota: int,
    state: Mapping[tuple[str, int], Mapping[str, float]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    adaptive = config["adaptive_sampling"]
    prior_alpha = float(adaptive["beta_prior_alpha"])
    prior_beta = float(adaptive["beta_prior_beta"])
    posterior = dict(
        state.get(
            (subcategory, difficulty),
            {
                "correct_count": 0,
                "incorrect_count": 0,
                "alpha": prior_alpha,
                "beta": prior_beta,
                "posterior_mean": prior_alpha / (prior_alpha + prior_beta),
                "posterior_variance": (
                    prior_alpha
                    * prior_beta
                    / (
                        (prior_alpha + prior_beta) ** 2
                        * (prior_alpha + prior_beta + 1)
                    )
                ),
                "observation_count": 0,
            },
        )
    )
    mean = float(posterior["posterior_mean"])
    variance = float(posterior["posterior_variance"])
    midpoint = float(adaptive["target_accuracy_mid"])
    temperature = max(float(adaptive["temperature"]), 1e-9)
    boundary = math.exp(-abs(mean - midpoint) / temperature)
    coverage_deficit = max(0.0, min_quota - current_count) / max(min_quota, 1)
    uncertainty = min(1.0, math.sqrt(max(variance, 0.0)) * 4)
    persistent_error = 1.0 - mean
    retention = mean
    score = (
        float(adaptive["boundary_weight"]) * boundary
        + float(adaptive["coverage_weight"]) * coverage_deficit
        + float(adaptive["uncertainty_weight"]) * uncertainty
        + float(adaptive["persistent_error_weight"]) * persistent_error
        + float(adaptive["retention_weight"]) * retention
    )
    observations = int(posterior["observation_count"])
    selected_difficulty = difficulty
    reasons = []
    minimum = int(adaptive["min_observations_before_adjustment"])
    if observations < minimum:
        reasons.append("insufficient_observations_explore")
    elif mean > float(adaptive["target_accuracy_high"]):
        selected_difficulty = min(10, difficulty + 1)
        reasons.append("above_target_accuracy_increase_difficulty")
    elif mean < float(adaptive["target_accuracy_low"]):
        selected_difficulty = max(1, difficulty - 1)
        reasons.append("below_target_accuracy_reduce_difficulty")
    else:
        reasons.append("inside_target_accuracy_range")
    if coverage_deficit > 0:
        reasons.append("coverage_quota_not_satisfied")
    if persistent_error > 0.5:
        reasons.append("persistent_error")
    return {
        "subcategory": subcategory,
        "previous_difficulty": difficulty,
        "selected_difficulty": selected_difficulty,
        "posterior_accuracy": mean,
        "posterior_variance": variance,
        "priority_score": score,
        "sampling_reason": reasons,
        **posterior,
    }


def generation_schedule(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    global_iteration: int,
    hard_pool_size: int,
    hard_pool_records: Iterable[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    records = list(records)
    items = taxonomy_items(config)
    metrics = coverage_metrics(records, config)
    state = beta_binomial_state(records, config)
    priorities = []
    for category, subcategory, metadata in items:
        difficulty = int(metadata.get("difficulty", 5))
        item = adaptive_priority(
            subcategory,
            difficulty,
            int(metrics["subcategory_counts"][subcategory]),
            int(metadata["min_quota"]),
            state,
            config,
        )
        item["category"] = category
        item["base_weight"] = float(metadata.get("base_weight", 1.0))
        item["current_count"] = int(metrics["subcategory_counts"][subcategory])
        item["min_quota"] = int(metadata["min_quota"])
        priorities.append(item)
    budget = int(config["experiment"]["questions_per_iteration"])
    injection_start = int(config["hard_pool"]["injection_start_iteration"])
    injection_enabled = global_iteration >= injection_start and hard_pool_size > 0
    if injection_enabled:
        source_budget = largest_remainder(
            budget,
            {
                "hard_pool_variant": float(
                    config["generation_mix"]["hard_pool_variants"]
                ),
                "coverage_deficit": float(
                    config["generation_mix"]["coverage_deficit"]
                ),
                "retention_known": float(
                    config["generation_mix"]["retention_known"]
                ),
            },
        )
    else:
        source_budget = {
            "hard_pool_variant": 0,
            "coverage_deficit": budget,
            "retention_known": 0,
        }
    hard_pool_records = list(hard_pool_records or [])
    eligible_hard_keys = {
        (
            str(record.get("category", "")),
            _record_subcategory(record),
        )
        for record in hard_pool_records
        if record.get("sample_grade") in {None, "train_eligible"}
    }
    if hard_pool_records and not eligible_hard_keys:
        injection_enabled = False
        source_budget = {
            "hard_pool_variant": 0,
            "coverage_deficit": budget,
            "retention_known": 0,
        }
    allocations = []
    actual_subcategory_budget: Counter[str] = Counter()
    max_questions_per_prompt = int(
        config["generation"]["max_questions_per_prompt"]
    )
    for source in (
        "hard_pool_variant",
        "coverage_deficit",
        "retention_known",
    ):
        source_total = source_budget[source]
        if source_total <= 0:
            continue
        source_items = priorities
        if source == "hard_pool_variant" and eligible_hard_keys:
            source_items = [
                item
                for item in priorities
                if (item["category"], item["subcategory"]) in eligible_hard_keys
            ]
        source_weights = {
            f"{item['category']}|||{item['subcategory']}": max(
                1e-9,
                item["priority_score"] * item["base_weight"],
            )
            for item in source_items
        }
        source_allocation = largest_remainder(source_total, source_weights)
        by_key = {
            f"{item['category']}|||{item['subcategory']}": item
            for item in source_items
        }
        for key, source_count in source_allocation.items():
            if not source_count:
                continue
            item = by_key[key]
            actual_subcategory_budget[item["subcategory"]] += source_count
            remaining_count = source_count
            while remaining_count:
                chunk_count = min(remaining_count, max_questions_per_prompt)
                allocations.append(
                    {
                        "category": item["category"],
                        "sub_category": item["subcategory"],
                        "question_count": chunk_count,
                        "difficulty": item["selected_difficulty"],
                        "generation_source": source,
                        "generation_strategy": (
                            "numeric_structure_variant"
                            if source == "hard_pool_variant"
                            else (
                                "quota_repair"
                                if source == "coverage_deficit"
                                else "retention_probe"
                            )
                        ),
                        "priority_score": item["priority_score"],
                    }
                )
                remaining_count -= chunk_count
    required_minimum = sum(item["min_quota"] for item in priorities)
    unsatisfied = [
        item["subcategory"]
        for item in priorities
        if item["current_count"] + actual_subcategory_budget[item["subcategory"]]
        < item["min_quota"]
    ]
    if sum(item["question_count"] for item in allocations) != budget:
        raise RuntimeError("generation allocation did not conserve the question budget")
    if global_iteration < injection_start and source_budget["hard_pool_variant"] != 0:
        raise RuntimeError("hard pool injection occurred during warmup")
    if injection_enabled and source_budget["hard_pool_variant"] <= 0:
        raise RuntimeError("directed generation budget must be positive after injection")
    return {
        "global_iteration": global_iteration,
        "question_budget": budget,
        "quota_feasible": budget >= required_minimum,
        "minimum_questions_for_full_quota": required_minimum,
        "unsatisfied_subcategories": unsatisfied,
        "hard_pool_injection_enabled": injection_enabled,
        "hard_pool_reference_count": min(
            hard_pool_size,
            int(config["hard_pool"]["max_reference_samples"]),
        ) if injection_enabled else 0,
        "directed_generation_question_count": source_budget["hard_pool_variant"],
        "coverage_repair_question_count": source_budget["coverage_deficit"],
        "retention_question_count": source_budget["retention_known"],
        "source_budget": source_budget,
        "allocations": allocations,
        "adaptive_sampler_state": priorities,
        "coverage_before_generation": metrics,
        "fallbacks": [],
    }
