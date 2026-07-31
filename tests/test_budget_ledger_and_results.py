from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from autobencher.budget_ledger import BudgetExhausted, BudgetLedger
from autobencher.config import ConfigurationError, load_resolved_config
from autobencher.paper_results import build_paper_tables
from autobencher.result_schema import ExperimentRecord
from autobencher.statistics import (
    holm_correction,
    mcnemar_test,
    paired_bootstrap,
)


ROOT = Path(__file__).resolve().parents[1]


def _ledger_config(protocol="question_matched"):
    return {
        "protocol": protocol,
        "data_matched_target_samples": 4 if protocol == "data_matched" else None,
        "max_generation_tokens": 10 if protocol == "cost_matched" else None,
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
    assert payload["estimated_cost"]["complete"] is True


def test_cost_matched_stops_after_recorded_token_cap(tmp_path):
    ledger = BudgetLedger(
        tmp_path / "ledger.json",
        _ledger_config("cost_matched"),
        {"study_id": "cost"},
    )
    ledger.record_generation_call(
        input_tokens=6,
        output_tokens=4,
        wall_time_seconds=0.1,
    )
    with pytest.raises(BudgetExhausted, match="token budget"):
        ledger.assert_generation_available()
    assert ledger.data["protocol"]["exhausted"] is True


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


def _write_run(
    root: Path,
    record: ExperimentRecord,
    baseline_answers: list[bool],
    final_answers: list[bool],
) -> None:
    run_dir = root / "output_root" / "test_1"
    for cycle, answers in ((0, baseline_answers), (1, final_answers)):
        stage = "baseline" if cycle == 0 else "cycle_1"
        stage_dir = run_dir / "fixed_test" / stage
        stage_dir.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "question_id": f"q{index}",
                "category": "Algebra",
                "sub_category": "Linear Equations",
                "difficulty": index,
                "gold_answer": "1",
                "test_taker_response": "1" if correct else "0",
                "is_correct": correct,
                "primary_error_tag": None if correct else "calculation_error",
                "parsed_response": {"confidence": 0.8},
                "latency": 0.1,
                "input_tokens": 10,
                "output_tokens": 2,
            }
            for index, correct in enumerate(answers, start=1)
        ]
        (stage_dir / "fixed_math.compare_answers.json").write_text(
            json.dumps(rows),
            encoding="utf-8",
        )
    (run_dir / "experiment_summary.json").write_text(
        json.dumps(
            {
                "baseline_accuracy": sum(baseline_answers)
                / len(baseline_answers),
                "final_accuracy": sum(final_answers) / len(final_answers),
                "accuracy_delta": (
                    sum(final_answers) / len(final_answers)
                    - sum(baseline_answers) / len(baseline_answers)
                ),
            }
        ),
        encoding="utf-8",
    )
    ledger = BudgetLedger(
        run_dir / "budget_ledger.json",
        _ledger_config(),
        {"study_id": record.study_id},
    )
    ledger.record_accuracy(
        sum(baseline_answers) / len(baseline_answers),
        sum(final_answers) / len(final_answers),
    )
    ledger.finalize("completed")


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
            config_hash=method,
            git_commit="commit",
            status="completed",
            experiment_dir=str(experiment_dir),
        )
        _write_run(
            experiment_dir,
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
                "experiments": [record.to_dict() for record in records],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "results"
    report = build_paper_tables(index, output)
    assert report["raw_item_rows"] == 12
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
    assert len(long_rows) == 12
    with (output / "significance_tests.csv").open(
        "r", encoding="utf-8", newline=""
    ) as handle:
        significance = list(csv.DictReader(handle))
    assert significance[0]["method"] == "full"
    assert "holm_adjusted_p_value" in significance[0]
