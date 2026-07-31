from __future__ import annotations

import json

import numpy as np
import pytest

from autobencher.config import load_resolved_config
from autobencher.coverage import beta_binomial_state, history_weight
from autobencher.difficulty_calibration import (
    DifficultyCalibrationError,
    calibrate_difficulty,
    fit_irt,
    freeze_calibration,
    prepare_panel_schedule,
)


def test_rasch_recovers_simulated_item_order_with_one_location_constraint():
    rng = np.random.default_rng(20260731)
    true_ability = np.linspace(-1.2, 2.6, 80)
    true_difficulty = np.linspace(-1.8, 1.8, 24)
    probability = 1.0 / (
        1.0
        + np.exp(
            -(true_ability[:, None] - true_difficulty[None, :])
        )
    )
    matrix = (rng.random(probability.shape) < probability).astype(float)
    fitted = fit_irt(matrix, iterations=3000, learning_rate=0.05)
    recovered = np.asarray(fitted["item_difficulty"])
    assert np.corrcoef(recovered, true_difficulty)[0, 1] > 0.9
    assert recovered.mean() == pytest.approx(0.0, abs=1.0e-10)
    # The panel is deliberately stronger than the zero-difficulty item mean;
    # its ability mean must not be artificially recentered to zero.
    assert np.mean(fitted["ability"]) > 0.2
    assert fitted["final_loss"] < fitted["loss_history"][0]["loss"]
    assert fitted["identification_constraint"] == "mean_item_difficulty_zero"


def _config(root, mode):
    config, _ = load_resolved_config(
        root / "configs" / "studies" / "full.yaml",
        temporary_overrides=[
            f"adaptive_sampling.history_mode={mode}",
            "adaptive_sampling.decay_lambda=0.5",
        ],
    )
    return config


def test_history_modes_have_explicit_model_version_semantics():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    old = {"cycle": 1, "sub_category": "Linear Equations", "difficulty": 4}
    assert history_weight(old, _config(root, "cumulative"), current_cycle=3) == 1
    assert history_weight(old, _config(root, "cycle_reset"), current_cycle=3) == 0
    assert history_weight(
        old, _config(root, "time_decay"), current_cycle=3
    ) == pytest.approx(2.718281828459045 ** -1)


def test_time_decay_changes_effective_not_raw_posterior_count():
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    config = _config(root, "time_decay")
    records = [
        {
            "cycle": 1,
            "sub_category": "Linear Equations",
            "difficulty": 4,
            "is_correct": False,
        },
        {
            "cycle": 3,
            "sub_category": "Linear Equations",
            "difficulty": 4,
            "is_correct": True,
        },
    ]
    state = beta_binomial_state(records, config, current_cycle=3)[
        ("Linear Equations", 4)
    ]
    assert state["raw_observation_count"] == 2
    assert state["observation_count"] == pytest.approx(1 + 2.718281828459045 ** -1)


def test_difficulty_calibration_rejects_blind_data_and_freezes(tmp_path):
    items = [
        {
            "question_id": f"q{index}",
            "difficulty": index + 1,
            "difficulty_profile": {
                "dimensions": {
                    "reasoning_steps": {"normalized": index / 4},
                    "operation_count": {"normalized": index / 4},
                    "constraint_count": {"normalized": index / 4},
                    "symbolic_depth": {"normalized": index / 4},
                    "representation_load": {"normalized": index / 4},
                }
            },
        }
        for index in range(5)
    ]
    # The implementation accepts scalar dimension values too; use those here
    # so this fixture stays independent of the live profile schema.
    for index, item in enumerate(items):
        item["dimensions"] = {
            name: float(index)
            for name in (
                "reasoning_steps",
                "operation_count",
                "constraint_count",
                "symbolic_depth",
                "representation_load",
            )
        }
        item.pop("difficulty_profile")
    panel = [
        {"model_id": "small", "tier": "small", "model_path": "/models/small"},
        {"model_id": "medium", "tier": "medium", "model_path": "/models/medium"},
        {"model_id": "strong", "tier": "strong", "model_path": "/models/strong"},
    ]
    with pytest.raises(DifficultyCalibrationError, match="blind"):
        prepare_panel_schedule(items, panel, dataset_role="blind_test")
    schedule = prepare_panel_schedule(
        items, panel, dataset_role="difficulty_calibration"
    )
    assert len(schedule) == 15
    responses = []
    for item_index, item in enumerate(items):
        for model_index, model in enumerate(panel):
            responses.append(
                {
                    "question_id": item["question_id"],
                    "model_id": model["model_id"],
                    "model_tier": model["tier"],
                    "is_correct": model_index >= item_index - 1,
                }
            )
    candidate = calibrate_difficulty(
        items,
        responses,
        dataset_role="difficulty_calibration",
        dataset_hash="dataset-hash",
        panel_hash="panel-hash",
    )
    assert candidate["rubric_version"] == "calibrated_math_v2"
    assert sum(candidate["weights"].values()) == pytest.approx(1.0)
    output = tmp_path / "calibrated_math_v2.json"
    frozen = freeze_calibration(candidate, output)
    assert frozen["status"] == "frozen"
    assert json.loads(output.read_text(encoding="utf-8"))["content_sha256"]
