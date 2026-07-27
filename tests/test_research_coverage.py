from pathlib import Path

import pytest

from autobencher.config import load_resolved_config
from autobencher.coverage import (
    adaptive_priority,
    beta_binomial_state,
    coverage_metrics,
    generation_schedule,
    largest_remainder,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "math_flywheel_smoke_test.yaml"
)


@pytest.fixture
def config():
    return load_resolved_config(CONFIG_PATH)[0]


def test_largest_remainder_conserves_budget():
    allocation = largest_remainder(27, {"a": 0.45, "b": 0.40, "c": 0.15})
    assert allocation == {"a": 12, "b": 11, "c": 4}
    assert sum(allocation.values()) == 27


def test_warmup_has_no_hard_pool_injection(config):
    schedule = generation_schedule([], config, global_iteration=1, hard_pool_size=9)
    assert schedule["source_budget"]["hard_pool_variant"] == 0
    assert schedule["question_budget"] == 27


def test_injection_uses_configured_mix(config):
    schedule = generation_schedule([], config, global_iteration=3, hard_pool_size=9)
    assert schedule["hard_pool_injection_enabled"] is True
    assert schedule["source_budget"] == {
        "hard_pool_variant": 12,
        "coverage_deficit": 11,
        "retention_known": 4,
    }


def test_iteration_three_falls_back_when_hard_pool_is_empty(config):
    schedule = generation_schedule([], config, global_iteration=3, hard_pool_size=0)
    assert schedule["hard_pool_injection_enabled"] is False
    assert schedule["source_budget"]["coverage_deficit"] == 27


def test_hard_variants_target_only_eligible_subcategories(config):
    hard_pool = [
        {
            "category": "Algebra",
            "sub_category": "Linear Equations",
            "sample_grade": "train_eligible",
        }
    ]
    schedule = generation_schedule(
        [],
        config,
        global_iteration=3,
        hard_pool_size=1,
        hard_pool_records=hard_pool,
    )
    hard_allocations = [
        item
        for item in schedule["allocations"]
        if item["generation_source"] == "hard_pool_variant"
    ]
    assert sum(item["question_count"] for item in hard_allocations) == 12
    assert {
        (item["category"], item["sub_category"])
        for item in hard_allocations
    } == {("Algebra", "Linear Equations")}


def test_schedule_allocations_conserve_exact_budget(config):
    schedule = generation_schedule([], config, global_iteration=1, hard_pool_size=0)
    assert sum(item["question_count"] for item in schedule["allocations"]) == 27
    assert schedule["quota_feasible"] is True


def test_balanced_coverage_is_complete_and_high_entropy(config):
    records = [
        {"sub_category": subcategory}
        for subcategories in config["taxonomy"].values()
        for subcategory in subcategories
    ]
    metrics = coverage_metrics(records, config)
    assert metrics["raw_subcategory_coverage"] == 1.0
    assert metrics["effective_subcategory_coverage"] == 1.0
    assert metrics["normalized_entropy"] == pytest.approx(1.0)
    assert metrics["js_divergence_to_uniform"] == pytest.approx(0.0)


def test_empty_coverage_is_zero(config):
    metrics = coverage_metrics([], config)
    assert metrics["raw_subcategory_coverage"] == 0.0
    assert metrics["effective_subcategory_coverage"] == 0.0
    assert metrics["normalized_entropy"] == 0.0


def test_beta_binomial_posterior_counts(config):
    records = [
        {"sub_category": "Integer Operations", "difficulty": 5, "is_correct": True},
        {"sub_category": "Integer Operations", "difficulty": 5, "is_correct": False},
    ]
    state = beta_binomial_state(records, config)
    posterior = state[("Integer Operations", 5)]
    assert posterior["alpha"] == 2
    assert posterior["beta"] == 2
    assert posterior["posterior_mean"] == 0.5


def test_adaptive_sampler_increases_easy_bucket_difficulty(config):
    state = {
        ("Integer Operations", 5): {
            "correct_count": 8,
            "incorrect_count": 0,
            "alpha": 9.0,
            "beta": 1.0,
            "posterior_mean": 0.9,
            "posterior_variance": 0.008,
            "observation_count": 8,
        }
    }
    priority = adaptive_priority(
        "Integer Operations", 5, 1, 1, state, config
    )
    assert priority["selected_difficulty"] == 6
    assert "above_target_accuracy_increase_difficulty" in priority["sampling_reason"]
