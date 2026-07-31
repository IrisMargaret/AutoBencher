from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from unittest.mock import Mock

import util

from autobencher.budget_ledger import (
    BudgetExhausted,
    BudgetLedger,
    set_active_ledger,
)
from autobencher.config import ConfigurationError, load_resolved_config
from autobencher.paper_results import build_paper_tables
from autobencher.result_schema import ExperimentRecord
from autobencher.statistics import (
    holm_correction,
    mcnemar_test,
    paired_bootstrap,
    summarize_values,
)
from artifact_fixtures import write_completed_run


ROOT = Path(__file__).resolve().parents[1]


def _ledger_config(protocol="question_matched"):
    return {
        "protocol": protocol,
        "data_matched_target_samples": 4 if protocol == "data_matched" else None,
        "max_generation_tokens": 10 if protocol == "generation_token_matched" else None,
        "max_total_api_calls": None,
        "max_gpu_hours": None,
        "pricing": {
            "currency": "USD",
            "input_per_million_tokens": 1.0,
            "output_per_million_tokens": 2.0,
            "gpu_hour": 3.0,
        },
    }


def test_budget_ledger_records_all_cost_sections_and_efficiency(tmp_path):
    path = tmp_path / "budget_ledger.json"
    ledger = BudgetLedger(path, _ledger_config(), {"study_id": "s1"})
    ledger.record_generation_call(
        input_tokens=100,
        output_tokens=50,
        wall_time_seconds=2.0,
        api_calls=1,
    )
    ledger.record_generation_batch(
        requested=10,
        output=9,
        accepted=8,
        failed=1,
        repair_retries=2,
        difficulty_rejections=1,
        sympy_validations=9,
        validation_failures=1,
    )
    records = [
        {"instruction": "solve", "input": "1+1", "output": "2"},
        {"instruction": "solve", "input": "2+2", "output": "4"},
    ]
    ledger.record_dataset(
        {
            "rejection_reasons": {
                "exact_duplicate": 2,
                "holdout_semantic_duplicate": 1,
            }
        },
        records,
    )
    ledger.record_training(
        {
            "training_token_count": 300,
            "optimizer_steps": 12,
            "gpu_hours": 0.5,
            "peak_gpu_memory_gb": 4.5,
            "training_wall_time_seconds": 1800,
        }
    )
    ledger.record_accuracy(0.2, 0.3)
    payload = ledger.finalize("completed")
    assert path.is_file()
    assert payload["generation"]["requested_question_count"] == 10
    assert payload["validation"]["sympy_validation_count"] == 9
    assert payload["validation"]["dedup_rejection_count"] == 2
    assert payload["validation"]["semantic_filter_count"] == 1
    assert payload["training"]["optimizer_steps"] == 12
    assert payload["totals"]["total_gpu_hours"] == 0.5
    assert payload["efficiency"][
        "accepted_samples_per_1k_generated_tokens"
    ] == pytest.approx(8 / 150 * 1000)
    assert payload["efficiency"][
        "accuracy_gain_per_1k_training_tokens"
    ] == pytest.approx(0.1 / 300 * 1000)
    assert payload["estimated_cost"]["complete"] is False
    assert payload["estimated_cost"]["cost_quality"] == "estimated"


def test_generation_token_matched_reserves_before_provider_call(tmp_path):
    ledger = BudgetLedger(
        tmp_path / "ledger.json",
        _ledger_config("generation_token_matched"),
        {"study_id": "cost"},
    )
    with pytest.raises(BudgetExhausted, match="token reservation exceeds budget"):
        ledger.assert_generation_available(
            reserved_input_tokens=6,
            reserved_output_tokens=5,
        )
    assert ledger.data["protocol"]["exhausted"] is True


def test_each_real_validation_provider_retry_is_recorded(tmp_path):
    ledger = BudgetLedger(
        tmp_path / "ledger.json", _ledger_config(), {"study_id": "judge"}
    )
    client = Mock()
    completion = Mock()
    completion.choices = [Mock(message=Mock(content='{"ok": true}'))]
    completion.usage = Mock(prompt_tokens=7, completion_tokens=4)
    client.chat.completions.create.side_effect = [RuntimeError("transient"), completion]
    set_active_ledger(ledger)
    try:
        result = util.query_openai_compatible(
            client,
            "judge",
            ["compare"],
            0.0,
            32,
            1,
            False,
            max_num_retries=2,
            retry_delay_seconds=0,
            budget_role="validation",
        )
    finally:
        set_active_ledger(None)
    assert result == ['{"ok": true}']
    assert ledger.data["validation"]["api_call_count"] == 2
    assert ledger.data["validation"]["judge_call_count"] == 2
    assert ledger.data["validation"]["validation_failure_count"] == 1
    assert ledger.data["validation"]["token_count_exact_calls"] == 1
    assert ledger.data["validation"]["token_count_estimated_calls"] == 1
    assert ledger.data["cost_quality"] == "mixed"


def test_resumed_ledger_cycle_events_are_idempotent(tmp_path):
    path = tmp_path / "budget_ledger.json"
    metadata = {
        "run_id": "resume",
        "config_hash": "config",
        "git_commit": "commit",
    }
    first = BudgetLedger(path, _ledger_config(), metadata)
    records = [{"instruction": "solve", "input": "1+1", "output": "2"}]
    assert first.record_dataset({}, records, event_id="cycle_1") is True
    assert first.record_training(
        {"final_training_sample_count": 1, "training_token_count": 3},
        event_id="cycle_1",
    ) is True
    resumed = BudgetLedger(
        path, _ledger_config(), metadata, resume=True
    )
    assert resumed.record_dataset({}, records, event_id="cycle_1") is False
    assert resumed.record_training(
        {"final_training_sample_count": 1, "training_token_count": 3},
        event_id="cycle_1",
    ) is False
    assert resumed.data["training"]["selected_pool_count"] == 1
    assert resumed.data["training"]["actual_trained_count"] == 1


def test_budget_protocol_validation_is_fail_closed():
    with pytest.raises(ConfigurationError, match="positive integer"):
        load_resolved_config(
            ROOT / "configs" / "studies" / "full.yaml",
            temporary_overrides=[
                "budget.protocol=data_matched",
                "budget.data_matched_target_samples=null",
            ],
        )
    with pytest.raises(ConfigurationError, match="divisible by 4"):
        load_resolved_config(
            ROOT / "configs" / "studies" / "full.yaml",
            temporary_overrides=[
                "budget.protocol=data_matched",
                "budget.data_matched_target_samples=5",
            ],
        )


def test_statistics_are_deterministic_and_report_effect_size():
    pairs = [(True, False), (False, True), (False, True), (True, True)]
    result = mcnemar_test(pairs)
    assert result["risk_difference"] == pytest.approx(0.25)
    assert result["matched_odds_ratio"] > 1
    assert paired_bootstrap(pairs, seed=7) == paired_bootstrap(pairs, seed=7)
    adjusted = holm_correction([0.01, 0.04, 0.03])
    assert all(0 <= value <= 1 for value in adjusted)


def test_three_seed_interval_uses_student_t_not_normal_critical_value():
    result = summarize_values([0.1, 0.2, 0.3])
    assert result["ci_method"] == "student_t_across_seeds"
    assert result["critical_value"] == pytest.approx(4.303, abs=0.001)
    expected_margin = 4.303 * 0.1 / (3 ** 0.5)
    assert result["ci95_low"] == pytest.approx(0.2 - expected_margin, abs=1e-4)
    assert result["ci95_high"] == pytest.approx(0.2 + expected_margin, abs=1e-4)


def test_paper_tables_rebuild_from_raw_item_artifacts(tmp_path):
    records = []
    for method, final in (
        ("base", [True, False, False]),
        ("full", [True, True, True]),
    ):
        experiment_dir = tmp_path / method
        record = ExperimentRecord(
            study_id=f"s-{method}",
            method=method,
            variant=method,
            seed=42,
            model="model",
            budget=30,
            config_hash="placeholder",
            git_commit="commit",
            status="completed",
            experiment_dir=str(experiment_dir),
        )
        write_completed_run(
            record,
            [True, False, False],
            final,
        )
        records.append(record)
    index = tmp_path / "experiment_index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "comparison_pairs": [{"left": "base", "right": "full"}],
                "experiments": [record.to_dict() for record in records],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "results"
    report = build_paper_tables(index, output)
    assert report["raw_item_rows"] == 9
    expected = {
        "results_long.csv",
        "run_summary.csv",
        "main_results.csv",
        "ablation_results.csv",
        "category_results.csv",
        "difficulty_results.csv",
        "efficiency_results.csv",
        "significance_tests.csv",
    }
    assert expected == {path.name for path in output.glob("*.csv")}
    with (output / "results_long.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        long_rows = list(csv.DictReader(handle))
    assert len(long_rows) == 9
    with (output / "significance_tests.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        significance = list(csv.DictReader(handle))
    assert significance[0]["method"] == "full"
    assert "holm_adjusted_p_value" in significance[0]
