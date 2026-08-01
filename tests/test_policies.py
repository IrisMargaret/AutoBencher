from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

import math_autobencher
from autobencher.config import load_resolved_config
from autobencher.coverage import generation_schedule
from autobencher.difficulty import assess_difficulty
from autobencher.experiment import ResearchRun
from autobencher.policies import (
    ErrorOnlyPolicy,
    FullAdaptivePolicy,
    RandomPolicy,
    UniformPolicy,
    create_policy,
)


ROOT = Path(__file__).resolve().parents[1]
SMOKE_CONFIG = ROOT / "configs" / "math_flywheel_smoke_test.yaml"
STUDY_ROOT = ROOT / "configs" / "studies"
STUDY_CONFIGS = {
    "random": STUDY_ROOT / "random.yaml",
    "uniform": STUDY_ROOT / "uniform.yaml",
    "error_only": STUDY_ROOT / "error_only.yaml",
    "full": STUDY_ROOT / "full.yaml",
    "full_no_hard_pool": (
        STUDY_ROOT / "ablations" / "full_no_hard_pool.yaml"
    ),
    "full_no_error_targeting": (
        STUDY_ROOT / "ablations" / "full_no_error_targeting.yaml"
    ),
    "full_no_observed_difficulty_sampling": (
        STUDY_ROOT
        / "ablations"
        / "full_no_observed_difficulty_sampling.yaml"
    ),
    "full_no_difficulty_module": (
        STUDY_ROOT / "ablations" / "full_no_difficulty_module.yaml"
    ),
}


def _config(method: str, *, budget: int = 53, maximum_chunk: int = 3):
    return load_resolved_config(
        STUDY_CONFIGS[method],
        temporary_overrides=[
            f"experiment.questions_per_iteration={budget}",
            f"generation.max_questions_per_prompt={maximum_chunk}",
        ],
    )[0]


def _schedule(config, *, seed=42, records=(), iteration=1, hard_pool=()):
    return generation_schedule(
        records,
        config,
        global_iteration=iteration,
        hard_pool_size=len(hard_pool),
        hard_pool_records=hard_pool,
        cycle=1,
        seed=seed,
    )


def _subcategory_counts(plan):
    counts = Counter()
    for allocation in plan["allocations"]:
        counts[
            (
                allocation["category"],
                allocation["sub_category"],
            )
        ] += allocation["question_count"]
    return counts


def test_random_same_seed_is_reproducible():
    config = _config("random")
    assert _schedule(config, seed=17) == _schedule(config, seed=17)
    assert isinstance(create_policy(config), RandomPolicy)


def test_random_different_seeds_change_the_plan():
    config = _config("random")
    assert _schedule(config, seed=17)["allocations"] != (
        _schedule(config, seed=18)["allocations"]
    )


def test_random_batches_by_the_27_configured_subcategories():
    config = _config("random", budget=90, maximum_chunk=50)
    plan = _schedule(config, seed=42)
    allocation_keys = [
        (item["category"], item["sub_category"])
        for item in plan["allocations"]
    ]
    assert len(allocation_keys) <= 27
    assert len(allocation_keys) == len(set(allocation_keys))
    assert sum(item["question_count"] for item in plan["allocations"]) == 90
    assert plan["diagnostics"]["batching_unit"] == "subcategory"
    assert plan["diagnostics"]["difficulty_sampling_unit"] == (
        "subcategory_batch"
    )


def test_uniform_subcategory_counts_differ_by_at_most_one():
    config = _config("uniform")
    plan = _schedule(config)
    counts = _subcategory_counts(plan)
    assert len(counts) == 27
    assert max(counts.values()) - min(counts.values()) <= 1
    assert {item["difficulty"] for item in plan["allocations"]} == {4}
    assert isinstance(create_policy(config), UniformPolicy)


@pytest.mark.parametrize("method", tuple(STUDY_CONFIGS))
def test_generation_policies_conserve_budget_and_chunk_limit(method):
    config = _config(method)
    plan = _schedule(config)
    assert sum(
        item["question_count"] for item in plan["allocations"]
    ) == plan["question_budget"] == 53
    assert sum(plan["source_budget"].values()) == 53
    assert max(item["question_count"] for item in plan["allocations"]) <= 3


def test_error_only_without_history_is_uniform():
    config = _config("error_only", budget=55)
    plan = _schedule(config)
    counts = _subcategory_counts(plan)
    assert max(counts.values()) - min(counts.values()) <= 1
    assert plan["diagnostics"]["fallback"] == "uniform_no_history"
    assert "error_only_no_history_uniform" in plan["fallbacks"]
    assert isinstance(create_policy(config), ErrorOnlyPolicy)


def test_error_only_gives_more_budget_to_higher_error_rate():
    config = _config("error_only", budget=270, maximum_chunk=50)
    records = [
        {
            "sub_category": "Integer Operations",
            "difficulty": 4,
            "is_correct": False,
        }
        for _ in range(20)
    ] + [
        {
            "sub_category": "Linear Equations",
            "difficulty": 4,
            "is_correct": True,
        }
        for _ in range(20)
    ]
    counts = _subcategory_counts(_schedule(config, records=records))
    assert counts[("Arithmetic", "Integer Operations")] > counts[
        ("Algebra", "Linear Equations")
    ]


def test_default_full_policy_preserves_golden_schedule():
    config = load_resolved_config(SMOKE_CONFIG)[0]
    warmup = _schedule(config, iteration=1)
    assert warmup["policy_name"] == "full"
    assert warmup["variant"] == "full"
    assert warmup["source_budget"] == {
        "hard_pool_variant": 0,
        "coverage_deficit": 27,
        "retention_known": 0,
    }
    assert len(warmup["allocations"]) == 27
    assert {
        (
            item["generation_source"],
            item["question_count"],
            item["difficulty"],
        )
        for item in warmup["allocations"]
    } == {("coverage_deficit", 1, 4)}

    hard_pool = (
        {
            "category": "Algebra",
            "sub_category": "Linear Equations",
            "sample_grade": "train_eligible",
        },
    )
    directed = _schedule(config, iteration=3, hard_pool=hard_pool)
    assert directed["source_budget"] == {
        "hard_pool_variant": 12,
        "coverage_deficit": 11,
        "retention_known": 4,
    }
    assert isinstance(create_policy(config), FullAdaptivePolicy)


def test_full_no_hard_pool_disables_and_redistributes_budget():
    config = _config("full_no_hard_pool", budget=27, maximum_chunk=50)
    hard_pool = (
        {
            "category": "Algebra",
            "sub_category": "Linear Equations",
            "sample_grade": "train_eligible",
        },
    )
    plan = _schedule(config, iteration=3, hard_pool=hard_pool)
    assert plan["hard_pool_injection_enabled"] is False
    assert plan["source_budget"] == {
        "hard_pool_variant": 0,
        "coverage_deficit": 20,
        "retention_known": 7,
    }
    assert sum(plan["source_budget"].values()) == 27
    assert plan["component_state"]["error_type_targeting"] is True
    assert plan["diagnostics"]["hard_pool_disabled_by_ablation"] is True


def test_full_no_observed_uses_requested_history_difficulty():
    config = _config("full_no_observed_difficulty_sampling", budget=27)
    records = [
        {
            "sub_category": "Integer Operations",
            "difficulty": 2,
            "target_difficulty": 6,
            "observed_difficulty": 2,
            "difficulty_profile": {
                "score": 2,
                "requested_score": 6,
                "effective_score": 2,
            },
            "is_correct": True,
        }
        for _ in range(8)
    ]
    plan = _schedule(config, records=records, iteration=2)
    state = next(
        item
        for item in plan["adaptive_sampler_state"]
        if item["subcategory"] == "Integer Operations"
    )
    assert state["previous_difficulty"] == 6
    assert plan["diagnostics"][
        "observed_difficulty_used_for_sampling"
    ] is False


def test_full_no_observed_still_records_observed_profile():
    config = _config("full_no_observed_difficulty_sampling")
    config["difficulty"]["minimum_profile_confidence"] = 0.0
    profile = assess_difficulty(
        "Solve for x: 2*x + 3 = 11.",
        "equation",
        {},
        ["Subtract 3.", "Divide by 2."],
        6,
        config,
    )
    assert "score" in profile
    assert profile["requested_score"] == 6
    assert profile["effective_score"] == 6
    assert profile["effective_score_source"] == "requested"


def test_hard_pool_and_error_targeting_are_independent_components():
    hard_off = _config("full_no_hard_pool")
    targeting_off = _config("full_no_error_targeting")
    assert hard_off["study"]["components"]["hard_pool_variants"] is False
    assert hard_off["study"]["components"]["error_type_targeting"] is True
    assert targeting_off["study"]["components"]["hard_pool_variants"] is True
    assert targeting_off["study"]["components"]["error_type_targeting"] is False


def test_full_no_difficulty_module_disables_every_difficulty_effect():
    config = _config("full_no_difficulty_module", budget=27)
    records = [
        {
            "sub_category": "Integer Operations",
            "difficulty": 9,
            "observed_difficulty": 9,
            "target_difficulty": 9,
            "cycle": 1,
            "is_correct": False,
        }
    ]
    plan = _schedule(config, records=records, iteration=2)
    assert plan["component_state"]["difficulty_module"] is False
    assert plan["component_evidence"]["difficulty_rejection_enabled"] is False
    assert plan["component_evidence"]["difficulty_mismatch_action"] == "disabled"
    assert {
        item["previous_difficulty"]
        for item in plan["adaptive_sampler_state"]
    } == {config["adaptive_sampling"]["initial_difficulty"]}
    assert all(
        check["passed"]
        for check in plan["diagnostics"]["component_runtime_checks"]
    )


def test_base_runs_fixed_evaluation_without_generation_or_training(
    tmp_path,
    monkeypatch,
):
    config, provenance = load_resolved_config(STUDY_ROOT / "base.yaml")
    config["paths"]["output_root"] = str(tmp_path)
    config["paths"]["enforce_data_root"] = False
    run = ResearchRun(config, provenance, "base-cpu", ROOT)
    run.initialize({"test": True})
    args = SimpleNamespace(
        research_run=run,
        outfile_prefix1=str(run.run_dir / "base."),
        mode="eval",
        agent_modelname="fixture-agent",
        test_taker_modelname="fixture-model",
        num_iters=5,
        max_cycle=3,
        export_interval=1,
        finetune_gpu="0",
        finetune_epoch=1,
        finetune_batch=1,
        lora_rank=8,
        use_helm="no",
    )
    fixed_metadata = {
        "sha256": "fixture-sha",
        "source_counts": {"fixture": 1},
    }
    monkeypatch.setattr(
        math_autobencher,
        "load_fixed_test_set",
        lambda *_args, **_kwargs: ([{"question_id": "fixed-1"}], fixed_metadata),
    )
    monkeypatch.setattr(
        math_autobencher,
        "_load_test_taker_info",
        lambda *_args, **_kwargs: ("fixture", None, None),
    )
    monkeypatch.setattr(
        math_autobencher,
        "_release_model_info",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        math_autobencher,
        "_run_fixed_test_benchmark",
        lambda *_args, **_kwargs: {
            "total_questions": 1,
            "accuracy": 0.5,
        },
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("base policy entered a generation/training stage")

    monkeypatch.setattr(math_autobencher, "_run_math_iteration", forbidden)
    monkeypatch.setattr(math_autobencher, "build_training_dataset", forbidden)
    monkeypatch.setattr(math_autobencher, "call_local_finetune", forbidden)

    assert math_autobencher._run_autobencher(args, None, None) == 0
    summary = json.loads(
        (run.run_dir / "experiment_summary.json").read_text(encoding="utf-8")
    )
    assert summary["policy_name"] == "base"
    assert summary["evaluation_only"] is True
    assert summary["iteration_count"] == 0
    assert summary["training_sample_count"] == 0
    assert not list((run.run_dir / "cycle").glob("cycle_*"))
