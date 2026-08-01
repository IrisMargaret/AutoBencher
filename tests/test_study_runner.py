from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

from aggregate_study import collect_results
from run_formal_evaluation import main as run_formal_evaluation
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
from artifact_fixtures import write_completed_run


ROOT = Path(__file__).resolve().parents[1]
STUDY_SUITE_DIR = ROOT / "configs" / "study_suites"
FIRST_ROUND_METHODS = [
    "base",
    "random",
    "uniform",
    "error_only",
    "full",
    "full_no_hard_pool",
    "full_no_error_targeting",
    "full_no_observed_difficulty_sampling",
    "full_no_difficulty_module",
]


def _load_study_suite(name: str) -> dict:
    payload = yaml.safe_load(
        (STUDY_SUITE_DIR / name).read_text(encoding="utf-8")
    )
    return payload["study_suite"]


def _resolved_full_config_for_suite(name: str) -> dict:
    suite = _load_study_suite(name)
    config, _ = load_resolved_config(
        ROOT / "configs" / "studies" / "full.yaml",
        ROOT / suite["environment"],
        temporary_overrides=suite.get("common_overrides", []),
        validate_paths=False,
    )
    return config


def test_three_strategy_single_cycle_suite_expands_to_nine_by_ninety():
    suite = _load_study_suite("three_strategy_single_cycle_810.yaml")
    assert suite["methods"] == ["full", "random", "uniform"]
    assert suite["comparison_pairs"] == [
        {"left": "random", "right": "full"},
        {"left": "uniform", "right": "full"},
    ]
    assert suite["seeds"] == [42]
    assert suite["budgets"] == [810]

    budget = BudgetSpec(
        total_questions=suite["budgets"][0],
        num_iterations=9,
        max_cycles=1,
    )
    assert budget.questions_per_iteration == 90
    assert budget.overrides() == [
        "experiment.num_iterations=9",
        "experiment.max_cycles=1",
        "experiment.questions_per_iteration=90",
    ]

    for method in suite["methods"]:
        config, _ = load_resolved_config(
            ROOT / "configs" / "studies" / f"{method}.yaml",
            ROOT / suite["environment"],
            temporary_overrides=suite["common_overrides"],
            validate_paths=False,
        )
        assert config["study"]["policy"] == method
        assert config["experiment"]["num_iterations"] == 9
        assert config["experiment"]["max_cycles"] == 1
        assert config["experiment"]["export_interval"] == 9
        assert config["finetune"]["enabled"] is True
        assert config["evaluator_pipeline"]["max_parallel_questions"] == 4


def test_observed_difficulty_ablation_is_preregistered_when_present():
    expected = {
        "left": "full_no_observed_difficulty_sampling",
        "right": "full",
    }
    checked = []
    for path in sorted(STUDY_SUITE_DIR.glob("*.yaml")):
        suite = _load_study_suite(path.name)
        if "full_no_observed_difficulty_sampling" not in suite["methods"]:
            continue
        checked.append(path.name)
        assert suite["comparison_pairs"].count(expected) == 1, path.name
    assert checked == [
        "budget_curve.yaml",
        "fair_budget.yaml",
        "main.yaml",
        "pilot.yaml",
        "smoke.yaml",
    ]


@pytest.mark.parametrize(
    "suite_name",
    ["main.yaml", "fair_budget.yaml", "ablation_round2.yaml"],
)
def test_research_suites_enable_retention_evaluation(suite_name):
    config = _resolved_full_config_for_suite(suite_name)
    assert config["retention_test"]["enabled"] is True


def test_smoke_suite_keeps_retention_evaluation_disabled():
    config = _resolved_full_config_for_suite("smoke.yaml")
    assert config["retention_test"]["enabled"] is False


def test_new_preregistered_pair_preserves_study_fairness(tmp_path):
    payload = yaml.safe_load(
        (STUDY_SUITE_DIR / "main.yaml").read_text(encoding="utf-8")
    )
    suite = payload["study_suite"]
    suite["name"] = "main_fairness_config_test"
    suite["output_root"] = str(tmp_path / "runs")
    suite["allow_missing_artifacts"] = True
    suite["allow_dirty_worktree"] = True
    suite["seeds"] = [42]
    suite["budgets"] = [135]
    suite["common_overrides"].append("paths.enforce_data_root=false")
    path = tmp_path / "main.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")

    plan = StudyRunner(path, project_root=ROOT).build_plan()
    assert {item.method for item in plan} == set(FIRST_ROUND_METHODS)
    assert len(plan) == len(FIRST_ROUND_METHODS)
    guidance_hashes = {
        item.fingerprints["generation_guidance"]["sha256"] for item in plan
    }
    assert len(guidance_hashes) == 1


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
            "methods": methods or FIRST_ROUND_METHODS,
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
        ("evaluator_solver.txt", "solver v1"),
        ("evaluator_independent_solver.txt", "independent v1"),
        ("evaluator_postcheck.txt", "postcheck v1"),
        ("tora_strategy.txt", "strategy v1"),
    ):
        (prompt_dir / name).write_text(text, encoding="utf-8")
    config = {
        "evaluator_pipeline": {
            "semantic_judge_prompt_path": "prompts/semantic.txt",
            "solver_prompt_path": "prompts/evaluator_solver.txt",
            "independent_solver_prompt_path": "prompts/evaluator_independent_solver.txt",
            "postcheck_prompt_path": "prompts/evaluator_postcheck.txt",
            "solver_strategy_path": "prompts/tora_strategy.txt",
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
    assert manifest["generation_guidance"]["sha256"]
    assert manifest["config"]["resolved_sha256"] == provenance["config_hash"]
    assert manifest["prompt_bundle"]["combined_sha256"]
    assert manifest["baseline_sha256"]


def test_first_round_plan_is_deterministic_and_isolated(tmp_path):
    suite = _write_suite(tmp_path)
    first = StudyRunner(suite, project_root=ROOT).build_plan()
    second = StudyRunner(suite, project_root=ROOT).build_plan()
    assert len(first) == 9
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
        "full_no_error_targeting",
        "full_no_observed_difficulty_sampling",
        "full_no_difficulty_module",
    }
    assert len({item.experiment_dir for item in first}) == 9
    for record in first:
        override_text = "\n".join(record.command)
        assert f"paths.cache_dir={record.experiment_dir}" in override_text
        assert f"paths.checkpoint_dir={record.experiment_dir}" in override_text
        assert record.budget == 5
        assert record.fingerprints["generation_guidance"]["kind"] == "disabled"


def test_return_code_zero_without_complete_artifacts_stays_partial(tmp_path):
    suite = _write_suite(tmp_path)
    calls = []

    def executor(command, cwd):
        calls.append((command, cwd))
        return 0

    runner = StudyRunner(suite, project_root=ROOT, executor=executor)
    first = runner.run(max_experiments=1)
    assert len(calls) == 1
    assert first[0].status == "partial"
    assert "Completion validation failed" in first[0].error

    resumed = StudyRunner(
        suite,
        project_root=ROOT,
        executor=executor,
    ).run(resume=True, max_experiments=1)
    assert len(calls) == 2
    assert resumed[0].status == "partial"
    with runner.index_path.open("r", encoding="utf-8") as handle:
        validate_registry(json.load(handle))


def test_formal_runner_binds_registry_method_seed_and_checkpoint(tmp_path, capsys):
    record = ExperimentRecord(
        study_id="source-full-42",
        method="full",
        variant="full",
        seed=42,
        model="fixture",
        budget=12,
        config_hash="placeholder",
        git_commit="commit",
        status="completed",
        experiment_dir=str(tmp_path / "experiment"),
    )
    write_completed_run(record, [False, True], [True, True])
    index = tmp_path / "index.json"
    index.write_text(
        json.dumps(
            {"schema_version": "1.0", "experiments": [record.to_dict()]}
        ),
        encoding="utf-8",
    )
    checkpoint_manifest = json.loads(
        next(
            Path(record.run_dir).glob(
                "cycle/cycle_*/training/checkpoint_manifest.json"
            )
        ).read_text(encoding="utf-8")
    )
    checkpoint = checkpoint_manifest["merged_model_path"]
    checkpoint_sha = checkpoint_manifest["merged_model_sha256"]
    assert run_formal_evaluation(
        [
            "--config", str(ROOT / "configs/experiments/official_fixed_eval.yaml"),
            "--source-index", str(index),
            "--source-study-id", record.study_id,
            "--source-method", "full",
            "--source-seed", "42",
            "--checkpoint-path", checkpoint,
            "--checkpoint-sha256", checkpoint_sha,
            "--run-id", "official-full-42",
            "--dry-run",
        ]
    ) == 0
    command = capsys.readouterr().out
    assert "evaluation_provenance.evaluated_method" in command
    assert "evaluation_provenance.source_run_id" in command
    assert checkpoint in command


def test_runner_expands_data_and_generation_token_protocols(tmp_path):
    suite = _write_suite(tmp_path)
    payload = yaml.safe_load(suite.read_text(encoding="utf-8"))
    payload["study_suite"]["protocols"] = [
        {"name": "data_matched", "target_training_samples": 4},
        {"name": "generation_token_matched", "max_generation_tokens": 1000},
    ]
    suite.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    plan = StudyRunner(suite, project_root=ROOT).build_plan()
    assert len(plan) == 18
    assert {item.budget_protocol for item in plan} == {
        "data_matched",
        "generation_token_matched",
    }
    assert len({item.experiment_dir for item in plan}) == 18


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


def test_aggregate_reads_registry_binding_not_latest_mtime(tmp_path):
    experiment_dir = tmp_path / "experiment"
    record = ExperimentRecord(
        study_id="study-aggregate",
        method="full",
        variant="full",
        seed=42,
        model="model",
        budget=10,
        config_hash="placeholder",
        git_commit="b",
        status="completed",
        experiment_dir=str(experiment_dir),
        run_dir=str(experiment_dir / "bound_run"),
    )
    write_completed_run(record, [True, False], [True, False])
    decoy = experiment_dir / "output_root" / "test_999"
    decoy.mkdir(parents=True)
    (decoy / "experiment_summary.json").write_text(
        json.dumps({"final_accuracy": 1.0}), encoding="utf-8"
    )
    index = tmp_path / "experiment_index.json"
    index.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "comparison_pairs": [],
                "experiments": [record.to_dict()],
            }
        ),
        encoding="utf-8",
    )
    results = collect_results(index)
    assert results[0]["final_accuracy"] == 0.5
    assert results[0]["accuracy_delta"] == 0.0
    assert results[0]["summary_path"] == record.summary_path


def test_study_runner_resume_reuses_same_directory_in_real_subprocess(tmp_path):
    suite = _write_suite(tmp_path, methods=["base"])
    runner = StudyRunner(suite, project_root=ROOT)
    planned = runner.build_plan()[0]
    command = planned.command
    config_path = Path(command[command.index("--config") + 1])
    environment = (
        Path(command[command.index("--environment") + 1])
        if "--environment" in command
        else None
    )
    overrides = command[command.index("--override") + 1 :]
    resolved, provenance = load_resolved_config(
        config_path,
        environment,
        temporary_overrides=overrides,
        validate_paths=False,
    )
    assert provenance["config_hash"] == planned.config_hash
    attempts = 0

    def executor(_command, _cwd):
        nonlocal attempts
        attempts += 1
        script = (
            "import pathlib,sys; p=pathlib.Path(sys.argv[1]); "
            "p.mkdir(parents=True,exist_ok=True); m=p/'child.marker'; "
            "existed=m.exists(); m.write_text('resumed' if existed else 'first'); "
            "sys.exit(0 if existed else 17)"
        )
        result = subprocess.run(
            [sys.executable, "-c", script, planned.run_dir],
            check=False,
        )
        if result.returncode == 0:
            write_completed_run(
                planned,
                [True, False, True],
                resolved_config=resolved,
            )
        return result.returncode

    runner.executor = executor
    first = runner.run(max_experiments=1)
    assert first[0].status == "partial"
    first_run_dir = first[0].run_dir
    resumed = StudyRunner(
        suite, project_root=ROOT, executor=executor
    ).run(resume=True, max_experiments=1)
    assert attempts == 2
    assert resumed[0].status == "completed"
    assert resumed[0].run_dir == first_run_dir
    assert (Path(first_run_dir) / "child.marker").read_text() == "resumed"
