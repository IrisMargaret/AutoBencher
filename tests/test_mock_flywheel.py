import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from autobencher.config import load_resolved_config
from autobencher.coverage import coverage_metrics, generation_schedule
from autobencher.dataset import build_training_dataset, write_alpaca_jsonl
from autobencher.experiment import ResearchRun
from autobencher.fingerprints import artifact_fingerprint
from autobencher.structured import answers_equivalent, parse_test_taker_output
from math_autobencher import (
    _failure_type,
    _load_resumable_cycle_checkpoint,
    _optional_accuracy_delta,
    _retention_delta,
    _run_fixed_test_benchmark,
    _sanitize_error,
    _sanitize_traceback,
    _upsert_history_iteration,
)


ROOT = Path(__file__).resolve().parents[1]


def test_cycle_history_upsert_is_idempotent():
    history = []
    first = [{"cycle_id": 1, "iteration_id": 1, "question_id": "q1"}]
    replacement = [
        {"cycle_id": 1, "iteration_id": 1, "question_id": "q1", "is_correct": True}
    ]
    _upsert_history_iteration(history, first, 1, 1)
    _upsert_history_iteration(history, replacement, 1, 1)
    assert history == [replacement]


def test_empty_exception_message_remains_diagnostic():
    try:
        raise AssertionError()
    except AssertionError as exc:
        assert _sanitize_error(exc) == "AssertionError()"
        assert "AssertionError" in _sanitize_traceback(exc)


def test_keyboard_interrupt_has_explicit_failure_type():
    assert _failure_type("evaluation", KeyboardInterrupt()) == "interrupted"


def test_optional_accuracy_delta_accepts_missing_baseline():
    assert _optional_accuracy_delta(None, 0.75) is None
    assert _optional_accuracy_delta("", 0.75) is None
    assert _optional_accuracy_delta(0.25, 0.75) == 0.5
    retention = _retention_delta(
        {"accuracy": None, "item_outcomes": []},
        {"accuracy": 0.75, "item_outcomes": []},
    )
    assert retention["accuracy_delta"] is None


def test_fixed_benchmark_resume_reuses_complete_stage_artifacts(
    tmp_path,
    monkeypatch,
):
    stage_dir = tmp_path / "fixed_test" / "cycle_1"
    stage_dir.mkdir(parents=True)
    questions = [
        {"question_id": "q1"},
        {"question_id": "q2"},
    ]
    model_name = "/models/cycle_1"
    summary = {
        "stage": "cycle_1",
        "model_name": model_name,
        "dataset_sha256": "fixed-sha",
        "total_questions": 2,
        "accuracy": 0.5,
        "item_outcomes": [
            {"question_id": "q1", "is_correct": True},
            {"question_id": "q2", "is_correct": False},
        ],
    }
    (stage_dir / "summary.json").write_text(
        json.dumps(summary),
        encoding="utf-8",
    )
    (stage_dir / "fixed_math.test_taker_inference.json").write_text(
        json.dumps([{"question_id": "q1"}, {"question_id": "q2"}]),
        encoding="utf-8",
    )
    (stage_dir / "fixed_math.compare_answers.json").write_text(
        json.dumps(
            {
                "total_questions": 2,
                "category_statistics": [],
            }
        ),
        encoding="utf-8",
    )

    logger = Mock()
    research_run = SimpleNamespace(run_dir=tmp_path, logger=logger)
    args = SimpleNamespace(research_run=research_run)

    def fail_if_recomputed(*_args, **_kwargs):
        raise AssertionError("completed fixed-test artifacts were recomputed")

    monkeypatch.setattr("math_autobencher.test_and_eval", fail_if_recomputed)
    resumed = _run_fixed_test_benchmark(
        args,
        model_name=model_name,
        test_taker_info=None,
        agent_info=None,
        evaluator_info=None,
        fixed_questions=questions,
        fixed_metadata={"sha256": "fixed-sha"},
        stage_name="cycle_1",
        cycle_number=1,
    )

    assert resumed == summary
    logger.event.assert_called_once_with(
        "INFO",
        "FixedTest",
        "stage_resume_hit",
        "stage=cycle_1 questions=2",
        cycle=1,
    )


def test_cycle_checkpoint_resume_reuses_verified_merged_model(tmp_path):
    model_dir = tmp_path / "models" / "cycle_1"
    model_dir.mkdir(parents=True)
    (model_dir / "config.json").write_text("{}", encoding="utf-8")
    (model_dir / "model.safetensors").write_bytes(b"merged-model")
    model_sha = artifact_fingerprint(
        model_dir,
        allow_missing=False,
    )["sha256"]
    checkpoint_path = tmp_path / "checkpoint_manifest.json"
    checkpoint = {
        "run_id": "same-run",
        "config_hash": "same-config",
        "base_model": "/models/base",
        "merged_model_path": str(model_dir),
        "merged_model_sha256": model_sha,
        "training_cost": {"optimizer_steps": 27},
    }
    checkpoint_path.write_text(json.dumps(checkpoint), encoding="utf-8")
    research_run = SimpleNamespace(
        run_id="same-run",
        config_hash="same-config",
    )

    recovered = _load_resumable_cycle_checkpoint(
        checkpoint_path,
        research_run,
        "/models/base",
    )

    assert recovered == checkpoint

    research_run.config_hash = "changed-config"
    with pytest.raises(RuntimeError, match="cannot be resumed"):
        _load_resumable_cycle_checkpoint(
            checkpoint_path,
            research_run,
            "/models/base",
        )


def test_cpu_mock_research_pipeline_exports_all_core_artifacts(tmp_path):
    config, provenance = load_resolved_config(
        ROOT / "configs" / "math_flywheel_smoke_test.yaml",
        temporary_overrides=[f"paths.output_root={json.dumps(str(tmp_path))}"],
    )
    run = ResearchRun(config, provenance, "cpu-mock", ROOT)
    run.initialize({"mock": True})
    schedule = generation_schedule([], config, global_iteration=1, hard_pool_size=0)
    questions = []
    outputs = []
    evaluations = []
    for index, allocation in enumerate(schedule["allocations"], start=1):
        answer = str(index + 1)
        question = {
            "question_id": f"mock-{index}",
            "question": f"What is {index} + 1?",
            "category": allocation["category"],
            "sub_category": allocation["sub_category"],
            "answer_type": "integer",
            "gold_answer": answer,
            "canonical_answer": answer,
            "gold_reasoning_summary": [
                f"Add {index} and 1 to obtain {answer}.",
                (
                    f"Check by subtracting 1 from {answer}; "
                    f"the original value {index} is recovered."
                ),
            ],
            "difficulty": allocation["difficulty"],
        }
        raw = json.dumps(
            {
                "reasoning_summary": ["Add one to the given integer."],
                "final_answer": answer,
                "answer_type": "integer",
                "confidence": 1.0,
            }
        )
        parsed = parse_test_taker_output(raw, None, "integer", config)
        equivalence = answers_equivalent(answer, answer, "integer", config)
        questions.append(question)
        outputs.append({**question, **parsed})
        evaluations.append(
            {
                **question,
                "test_taker_response": answer,
                "is_correct": equivalence["equivalent"],
                "parse_status": parsed["parse_status"],
                "sub_category_accuracy": 1.0,
                "evaluator_confidence": 1.0,
                "question_parse_success": True,
                "answer_validation_success": True,
            }
        )
    metrics = coverage_metrics(evaluations, config)
    run.export_iteration(
        1,
        1,
        {
            "generation_plan": schedule,
            "generated_questions": questions,
            "test_taker_outputs": outputs,
            "evaluation_results": evaluations,
            "coverage_metrics": metrics,
        },
    )
    alpaca, manifest, rejected = build_training_dataset(evaluations, config)
    training_dir = run.run_dir / "cycle" / "cycle_1" / "training"
    dataset_path = training_dir / "dataset.jsonl"
    write_alpaca_jsonl(alpaca, dataset_path)
    run.save_cycle_artifact(1, "training", "dataset_manifest", manifest)
    run.save_cycle_artifact(1, "training", "rejected_samples", rejected)
    assert metrics["raw_subcategory_coverage"] == 1.0
    assert dataset_path.is_file()
    assert (
        run.run_dir
        / "cycle"
        / "cycle_1"
        / "iter_1"
        / "evaluation_results.json"
    ).is_file()
