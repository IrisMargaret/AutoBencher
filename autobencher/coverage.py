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


def _record_is_correct(record: Mapping[str, Any]) -> bool:
    value = record.get("is_correct", False)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes"}


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
        correct = sum(_record_is_correct(item) for item in group)
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


def previous_round_accuracy_state(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Translate the immediately previous round's accuracy into a difficulty bias."""
    adaptive = config["adaptive_sampling"]
    evaluated = [record for record in records if "is_correct" in record]
    observations = len(evaluated)
    enabled = bool(adaptive["global_accuracy_enabled"])
    minimum = int(adaptive["global_accuracy_min_observations"])
    if not enabled:
        return {
            "enabled": False,
            "observation_count": observations,
            "accuracy": None,
            "band": "disabled",
            "difficulty_delta": 0,
            "reason": "global_accuracy_adjustment_disabled",
        }
    if observations < minimum:
        return {
            "enabled": True,
            "observation_count": observations,
            "accuracy": None,
            "band": "insufficient_observations",
            "difficulty_delta": 0,
            "reason": "global_accuracy_insufficient_observations",
        }
    accuracy = sum(_record_is_correct(record) for record in evaluated) / observations
    low = float(adaptive["global_accuracy_low"])
    high = float(adaptive["global_accuracy_high"])
    step = int(adaptive["global_difficulty_step"])
    if accuracy < low:
        band = "below_target"
        delta = -step
        reason = "previous_round_accuracy_low_reduce_difficulty"
    elif accuracy > high:
        band = "above_target"
        delta = step
        reason = "previous_round_accuracy_high_increase_difficulty"
    else:
        band = "inside_target"
        delta = 0
        reason = "previous_round_accuracy_inside_target_keep_difficulty"
    return {
        "enabled": True,
        "observation_count": observations,
        "accuracy": accuracy,
        "target_low": low,
        "target_high": high,
        "band": band,
        "difficulty_delta": delta,
        "reason": reason,
    }


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
    previous_round_records: Iterable[Mapping[str, Any]] | None = None,
    *,
    cycle: int = 0,
    seed: int | None = None,
    question_budget: int | None = None,
) -> dict[str, Any]:
    # Local import prevents a module cycle: policies intentionally reuse the
    # pure calculations defined above.
    from .policies import PolicyContext, create_policy

    context = PolicyContext(
        config=config,
        history_records=tuple(records),
        previous_round_records=tuple(previous_round_records or ()),
        hard_pool_records=tuple(hard_pool_records or ()),
        global_iteration=int(global_iteration),
        cycle=int(cycle),
        seed=(
            int(config["experiment"].get("seed", 42))
            if seed is None
            else int(seed)
        ),
        question_budget=(
            int(config["experiment"]["questions_per_iteration"])
            if question_budget is None
            else int(question_budget)
        ),
        hard_pool_size=int(hard_pool_size),
    )
    return create_policy(config).build_plan(context).to_dict()
