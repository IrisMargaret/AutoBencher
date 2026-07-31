from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml

from aggregate_study import collect_results
from autobencher.baseline import build_baseline_manifest
from autobencher.budget import BudgetSpec
from autobencher.config import load_resolved_config
from autobencher.fingerprints import prompt_bundle_snapshot
from autobencher.result_schema import (
    ExperimentRecord,
    validate_registry,
)
from autobencher.study_runner import (
    StudyConfigurationError,
    StudyRunner,
)


ROOT = Path(__file__).resolve().parents[1]
SEVEN_METHODS = [
    "base",
    "random",
    "uniform",
    "error_only",
    "full",
    "full_no_hard_pool",
    "full_no_observed_difficulty_sampling",
]


def _write_suite(
    tmp_path: Path,
    *,
    methods: list[str] | None = None,
    method_configs: dict[str, str] | None = None,
) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "config.json").write_text(
        '{"model_type":"test"}\n',
        encoding="utf-8",
    )
    payload = {
        "schema_version": "1.0",
        "study_suite": {
            "name": "test_suite",
            "output_root": str(tmp_path / "runs"),
            "allow_missing_artifacts": False,
            "allow_dirty_worktree": True,
            "methods": methods or SEVEN_METHODS,
            "seeds": [42],
            "models": [{"name": "test-model", "path": str(model_dir)}],
            "budgets": [5],
            "common_overrides": [
                "paths.enforce_data_root=false",
                "experiment.num_iterations=5",
                "experiment.max_cycles=1",
            ],
        },
    }
    if method_configs:
        payload["study_suite"]["method_configs"] = method_configs
    path = tmp_path / "suite.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def test_prompt_bundle_hashes_actual_file_contents(tmp_path):
    prompt_dir = tmp_path / "prompts"
    prompt_dir.mkdir()
    for name, text in (
        ("generator_question.txt", "generator v1"),
        ("test_taker.txt", "test taker v1"),
        ("semantic.txt", "semantic v1"),
    ):
        (prompt_dir / name).write_text(text, encoding="utf-8")
    config = {
        "evaluator_pipeline": {
            "semantic_judge_prompt_path": "prompts/semantic.txt"
        }
    }
    first = prompt_bundle_snapshot(config, tmp_path)
    (prompt_dir / "test_taker.txt").write_text(
        "test taker v2",
        encoding="utf-8",
    )
    second = prompt_bundle_snapshot(config, tmp_path)
    assert first["generator"] == second["generator"]
    assert first["test_taker"]["sha256"] != second["test_taker"]["sha256"]
    assert first["combined_sha256"] != second["combined_sha256"]


def test_budget_is_total_and_must_split_exactly():
    budget = BudgetSpec(
        total_questions=1350,
        num_iterations=5,
        max_cycles=3,
    )
    assert budget.questions_per_iteration == 90
    assert "experiment.questions_per_iteration=90" in budget.overrides()
    with pytest.raises(ValueError, match="divisible"):
        BudgetSpec(total_questions=8, num_iterations=3, max_cycles=2)


def test_baseline_manifest_records_required_fingerprints():
    config, provenance = load_resolved_config(
        ROOT / "configs" / "studies" / "full.yaml",
        validate_paths=False,
    )
    manifest = build_baseline_manifest(
        config,
        provenance,
        ROOT,
        allow_missing_artifacts=True,
    )
    assert manifest["git"]["commit"]
    assert manifest["environment"]["python"]
    assert "transformers" in manifest["environment"]["packages"]
    assert manifest["model"]["sha256"]
    assert manifest["fixed_test"]["sha256"]
    assert manifest["config"]["resolved_sha256"] == provenance["config_hash"]
    assert manifest["prompt_bundle"]["combined_sha256"]
    assert manifest["baseline_sha256"]


def test_seven_method_plan_is_deterministic_and_isolated(tmp_path):
    suite = _write_suite(tmp_path)
    first = StudyRunner(suite, project_root=ROOT).build_plan()
    second = StudyRunner(suite, project_root=ROOT).build_plan()
    assert len(first) == 7
    assert [item.study_id for item in first] == [
        item.study_id for item in second
    ]
    assert {item.method for item in first} == {
        "base",
        "random",
        "uniform",
        "error_only",
        "full",
        "full_no_hard_pool",
        "full_no_observed_difficulty",
    }
    assert len({item.experiment_dir for item in first}) == 7
    for record in first:
        override_text = "\n".join(record.command)
        assert f"paths.cache_dir={record.experiment_dir}" in override_text
        assert f"paths.checkpoint_dir={record.experiment_dir}" in override_text
        assert record.budget == 5


def test_registry_resume_skips_completed_and_continues_next(tmp_path):
    suite = _write_suite(tmp_path)
    calls = []

    def executor(command, cwd):
        calls.append((command, cwd))
        return 0

    runner = StudyRunner(suite, project_root=ROOT, executor=executor)
    first = runner.run(max_experiments=1)
    assert len(calls) == 1
    assert sum(item.status == "completed" for item in first) == 1

    resumed = StudyRunner(
        suite,
        project_root=ROOT,
        executor=executor,
    ).run(resume=True, max_experiments=1)
    assert len(calls) == 2
    assert sum(item.status == "completed" for item in resumed) == 2
    with runner.index_path.open("r", encoding="utf-8") as handle:
        validate_registry(json.load(handle))


def test_fairness_rejects_non_strategy_training_change(tmp_path):
    unfair_config = tmp_path / "unfair_random.yaml"
    unfair_config.write_text(
        "extends: "
        + (ROOT / "configs" / "studies" / "random.yaml").as_posix()
        + "\nfinetune:\n  epochs: 99\n",
        encoding="utf-8",
    )
    suite = _write_suite(
        tmp_path,
        methods=["random", "full"],
        method_configs={"random": str(unfair_config)},
    )
    with pytest.raises(StudyConfigurationError, match="Unfair"):
        StudyRunner(suite, project_root=ROOT).build_plan()


def test_experiment_record_requires_unique_valid_status():
    record = ExperimentRecord(
        study_id="study-1",
        method="full",
        variant="full",
        seed=42,
        model="model",
        budget=10,
        config_hash="a",
        git_commit="b",
    )
    payload = {
        "schema_version": "1.0",
        "experiments": [record.to_dict()],
    }
    validate_registry(payload)
    payload["experiments"].append(record.to_dict())
    with pytest.raises(ValueError, match="duplicate"):
        validate_registry(payload)


def test_aggregate_collects_latest_experiment_summary(tmp_path):
    experiment_dir = tmp_path / "experiment"
    first = experiment_dir / "output_root" / "test_1"
    second = experiment_dir / "output_root" / "test_2"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    first_summary = first / "experiment_summary.json"
    first_summary.write_text(
        json.dumps(
            {
                "baseline_accuracy": 0.2,
                "final_accuracy": 0.3,
                "accuracy_delta": 0.1,
            }
        ),
        encoding="utf-8",
    )
    latest = second / "experiment_summary.json"
    latest.write_text(
        json.dumps(
            {
                "baseline_accuracy": 0.2,
                "final_accuracy": 0.4,
                "accuracy_delta": 0.2,
            }
        ),
        encoding="utf-8",
    )
    os.utime(first_summary, (1, 1))
    os.utime(latest, (2, 2))
    record = ExperimentRecord(
        study_id="study-aggregate",
        method="full",
        variant="full",
        seed=42,
        model="model",
        budget=10,
        config_hash="a",
        git_commit="b",
        status="completed",
        experiment_dir=str(experiment_dir),
    )
    index = tmp_path / "experiment_index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "experiments": [record.to_dict()],
            }
        ),
        encoding="utf-8",
    )
    results = collect_results(index)
    assert results[0]["final_accuracy"] == 0.4
    assert results[0]["accuracy_delta"] == 0.2
    assert results[0]["summary_path"] == str(latest)
