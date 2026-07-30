from pathlib import Path

import pytest

from autobencher.config import load_resolved_config
from autobencher.coverage import generation_schedule
from autobencher.difficulty import (
    analyze_difficulty,
    assess_difficulty,
    target_difficulty_profile,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "math_flywheel_smoke_test.yaml"
)


@pytest.fixture
def config():
    return load_resolved_config(CONFIG_PATH)[0]


def test_target_profile_defines_work_not_large_numbers():
    profile = target_difficulty_profile(5)
    assert profile["band"] == "integrated"
    assert profile["expected_reasoning_steps"] == [3, 5]
    assert "large numbers" in profile["instruction"]


def test_large_literals_do_not_artificially_raise_difficulty():
    small = analyze_difficulty(
        "Compute 2 + 3.",
        "integer",
        reasoning_steps=["Add the two integers."],
    )
    large = analyze_difficulty(
        "Compute 987654321 + 123456789.",
        "integer",
        reasoning_steps=["Add the two integers."],
    )
    assert large["score"] == small["score"]
    assert (
        large["dimensions"]["operation_count"]["value"]
        == small["dimensions"]["operation_count"]["value"]
    )


def test_verified_multiconstraint_problem_scores_above_direct_arithmetic(
    config,
):
    direct = analyze_difficulty(
        "Compute 2 + 3.",
        "integer",
        {"normalized_expressions": ["2 + 3"]},
        ["Add 2 and 3.", "Check that 5 - 3 = 2."],
        config,
    )
    system = assess_difficulty(
        "Solve the system for (x, y): x + y = 7, 2*x - y = 2.",
        "ordered_tuple",
        {
            "normalized_expressions": [
                "x + y = 7",
                "2*x - y = 2",
            ]
        },
        [
            "Write both equations.",
            "Eliminate y from the system.",
            "Solve the resulting equation for x.",
            "Substitute x to obtain y.",
            "Check the pair in both equations.",
        ],
        4,
        config,
    )
    assert system["score"] > direct["score"]
    assert system["profile_trusted"] is True
    assert system["effective_score"] == system["score"]
    assert system["dimensions"]["constraint_count"]["value"] >= 2


def test_generation_schedule_exports_target_difficulty_profile(config):
    schedule = generation_schedule(
        [],
        config,
        global_iteration=1,
        hard_pool_size=0,
    )
    assert schedule["allocations"]
    assert all(
        allocation["target_difficulty_profile"]["target_score"]
        == allocation["difficulty"]
        for allocation in schedule["allocations"]
    )
    assert all(
        allocation["target_difficulty_profile"]["rubric_version"]
        == "observable_math_v1"
        for allocation in schedule["allocations"]
    )
