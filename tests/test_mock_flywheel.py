import json
from pathlib import Path

from autobencher.config import load_resolved_config
from autobencher.coverage import coverage_metrics, generation_schedule
from autobencher.dataset import build_training_dataset, write_alpaca_jsonl
from autobencher.experiment import ResearchRun
from autobencher.structured import answers_equivalent, parse_test_taker_output
from math_autobencher import (
    _failure_type,
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
                "Add one to the given integer and verify the sum."
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
