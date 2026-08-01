"""Deterministic, isolated, resumable execution of ablation-study matrices."""

from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from autobencher.baseline import git_state
from autobencher.budget import parse_budget
from autobencher.config import load_resolved_config, thaw_config
from autobencher.experiment import (
    atomic_json,
    git_commit,
    run_dir_for_id,
    utc_now,
)
from autobencher.fingerprints import (
    artifact_fingerprint,
    canonical_sha256,
    prompt_bundle_snapshot,
    file_sha256,
    resolve_project_path,
)
from autobencher.result_schema import (
    ExperimentRecord,
    validate_registry,
)


METHOD_CONFIGS = {
    "base": "configs/studies/base.yaml",
    "random": "configs/studies/random.yaml",
    "uniform": "configs/studies/uniform.yaml",
    "error_only": "configs/studies/error_only.yaml",
    "full": "configs/studies/full.yaml",
    "full_no_hard_pool": (
        "configs/studies/ablations/full_no_hard_pool.yaml"
    ),
    "full_no_error_targeting": (
        "configs/studies/ablations/full_no_error_targeting.yaml"
    ),
    "full_no_observed_difficulty_sampling": (
        "configs/studies/ablations/full_no_observed_difficulty_sampling.yaml"
    ),
    "full_no_difficulty_module": (
        "configs/studies/ablations/full_no_difficulty_module.yaml"
    ),
    "full_no_coverage_priority": (
        "configs/studies/ablations/full_no_coverage_priority.yaml"
    ),
    "full_no_uncertainty_priority": (
        "configs/studies/ablations/full_no_uncertainty_priority.yaml"
    ),
    "full_no_global_difficulty": (
        "configs/studies/ablations/full_no_global_difficulty.yaml"
    ),
    "full_no_retention_priority": (
        "configs/studies/ablations/full_no_retention_priority.yaml"
    ),
    "full_history_cumulative": (
        "configs/studies/history/full_history_cumulative.yaml"
    ),
    "full_history_cycle_reset": (
        "configs/studies/history/full_history_cycle_reset.yaml"
    ),
    "full_history_time_decay": (
        "configs/studies/history/full_history_time_decay.yaml"
    ),
}

METHOD_ALIASES = {
    "full_no_observed_difficulty": (
        "full_no_observed_difficulty_sampling"
    ),
}

ALLOWED_METHOD_SPECIFIC_FIELDS = {
    "adaptive_sampling.history_mode",
    "adaptive_sampling.decay_lambda",
}

RUNTIME_PATH_FIELDS = (
    "output_root",
    "cache_dir",
    "model_output_dir",
    "log_dir",
    "temp_dir",
    "dataset_dir",
    "checkpoint_dir",
    "review_dir",
)


class StudyConfigurationError(RuntimeError):
    pass


def _json_mapping(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StudyConfigurationError(f"Invalid required artifact {path}: {error}")
    if not isinstance(value, dict):
        raise StudyConfigurationError(f"Required artifact is not an object: {path}")
    return value


def _question_ids(path: Path) -> set[str]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise StudyConfigurationError(f"Expected item list: {path}")
    identifiers = {
        str(item.get("question_id", item.get("id", "")))
        for item in value
        if isinstance(item, Mapping)
    }
    if "" in identifiers or len(identifiers) != len(value):
        raise StudyConfigurationError(f"Missing or duplicate question IDs: {path}")
    return identifiers


def validate_experiment_completion(record: ExperimentRecord) -> dict[str, Any]:
    """Fail closed unless one registry-bound run is complete and self-consistent."""
    if not record.run_dir:
        raise StudyConfigurationError(f"Registry has no run_dir for {record.study_id}")
    run_dir = Path(record.run_dir).resolve()
    expected_paths = {
        "run_manifest": Path(record.run_manifest_path or run_dir / "run_manifest.json").resolve(),
        "summary": Path(record.summary_path or run_dir / "experiment_summary.json").resolve(),
        "artifact_manifest": Path(
            record.artifact_manifest_path or run_dir / "artifact_manifest.json"
        ).resolve(),
        "ledger": (run_dir / "budget_ledger.json").resolve(),
        "resolved_config": (run_dir / "resolved_config.json").resolve(),
        "dataset_snapshot": (run_dir / "fixed_test" / "dataset_snapshot.json").resolve(),
    }
    if any(not path.is_relative_to(run_dir) for path in expected_paths.values()):
        raise StudyConfigurationError("Registry artifact path escapes bound run_dir")
    missing = [name for name, path in expected_paths.items() if not path.is_file()]
    if missing:
        raise StudyConfigurationError(
            f"Incomplete experiment {record.study_id}; missing {missing}"
        )
    manifest = _json_mapping(expected_paths["run_manifest"])
    summary = _json_mapping(expected_paths["summary"])
    ledger = _json_mapping(expected_paths["ledger"])
    resolved = _json_mapping(expected_paths["resolved_config"])
    artifacts = _json_mapping(expected_paths["artifact_manifest"])
    expected_identity = {
        "run_id": record.study_id,
        "config_hash": record.config_hash,
        "git_commit": record.git_commit,
    }
    for name, payload in (("manifest", manifest), ("summary", summary), ("ledger", ledger)):
        for key, expected in expected_identity.items():
            if payload.get(key) != expected:
                raise StudyConfigurationError(
                    f"{name} identity mismatch for {key}: "
                    f"{payload.get(key)!r} != {expected!r}"
                )
    for key, expected in expected_identity.items():
        if artifacts.get(key) != expected:
            raise StudyConfigurationError(
                f"artifact manifest identity mismatch for {key}: "
                f"{artifacts.get(key)!r} != {expected!r}"
            )
    if manifest.get("status") != "completed" or summary.get("status") != "completed":
        raise StudyConfigurationError("Run manifest/summary is not completed")
    if ledger.get("status") != "completed":
        raise StudyConfigurationError("Budget ledger is not completed")
    if ledger.get("protocol", {}).get("name") != record.budget_protocol:
        raise StudyConfigurationError("Budget protocol differs from registry")
    if manifest.get("prompt_hash") != record.fingerprints["prompt_bundle"]["combined_sha256"]:
        raise StudyConfigurationError("Prompt fingerprint changed inside run")
    if manifest.get("base_model_sha256") != record.fingerprints["base_model"]["sha256"]:
        raise StudyConfigurationError("Base-model fingerprint changed inside run")
    guidance_sha256 = record.fingerprints.get("generation_guidance", {}).get(
        "sha256"
    )
    if not guidance_sha256 or manifest.get(
        "generation_guidance_sha256"
    ) != guidance_sha256:
        raise StudyConfigurationError(
            "Generation-guidance fingerprint changed inside run"
        )
    if canonical_sha256(resolved) != record.config_hash:
        raise StudyConfigurationError("Resolved configuration hash mismatch")
    artifact_entries = {
        str(item.get("path")): item for item in artifacts.get("files", [])
        if isinstance(item, Mapping)
    }
    required_manifest_entries = {
        path.relative_to(run_dir).as_posix()
        for name, path in expected_paths.items()
        if name != "artifact_manifest"
    }
    omitted = sorted(required_manifest_entries - set(artifact_entries))
    if omitted:
        raise StudyConfigurationError(
            f"Artifact manifest omits required files: {omitted}"
        )
    for item in artifact_entries.values():
        path = (run_dir / str(item.get("path", ""))).resolve()
        if not path.is_relative_to(run_dir) or not path.is_file():
            raise StudyConfigurationError(f"Artifact manifest path is invalid: {path}")
        if file_sha256(path) != item.get("sha256"):
            raise StudyConfigurationError(f"Artifact hash mismatch: {path}")
    snapshot = _json_mapping(expected_paths["dataset_snapshot"])
    fixed_sha = record.fingerprints.get("fixed_test", {}).get("sha256")
    if not fixed_sha or snapshot.get("sha256") != fixed_sha:
        raise StudyConfigurationError(
            "Fixed-test snapshot hash differs from the preregistered dataset"
        )
    questions = snapshot.get("questions", [])
    expected_ids = {
        str(item.get("question_id", item.get("id", "")))
        for item in questions
        if isinstance(item, Mapping)
    }
    if not expected_ids or "" in expected_ids or len(expected_ids) != len(questions):
        raise StudyConfigurationError("Fixed-test snapshot is incomplete")
    declared_count = snapshot.get("question_count")
    if declared_count is not None and int(declared_count) != len(expected_ids):
        raise StudyConfigurationError("Fixed-test snapshot count is inconsistent")
    # ``total_questions`` is the generation count for training runs. Fixed-set
    # completeness is tracked separately and ultimately enforced item-by-item
    # against every answer-comparison artifact below.
    summary_count = summary.get("fixed_test_question_count")
    if summary_count is not None and int(summary_count) != len(expected_ids):
        raise StudyConfigurationError("Experiment summary test count is incomplete")
    comparisons = sorted(
        (run_dir / "fixed_test").glob("*/fixed_math.compare_answers.json")
    )
    if not comparisons:
        raise StudyConfigurationError("No fixed-test answer comparison exists")
    for path in comparisons:
        if _question_ids(path) != expected_ids:
            raise StudyConfigurationError(
                f"Fixed-test item set is incomplete or inconsistent: {path}"
            )
        relative = path.relative_to(run_dir).as_posix()
        if relative not in artifact_entries:
            raise StudyConfigurationError(
                f"Artifact manifest omits fixed-test result: {relative}"
            )
    is_base = record.method == "base"
    checkpoint_sha = record.fingerprints["base_model"]["sha256"]
    if not is_base:
        checkpoint_manifests = sorted(
            run_dir.glob("cycle/cycle_*/training/checkpoint_manifest.json")
        )
        if not checkpoint_manifests:
            raise StudyConfigurationError("Training run has no checkpoint manifest")
        checkpoint = _json_mapping(checkpoint_manifests[-1])
        for key, expected in expected_identity.items():
            if checkpoint.get(key) != expected:
                raise StudyConfigurationError(
                    f"Checkpoint identity mismatch for {key}"
                )
        checkpoint_relative = checkpoint_manifests[-1].relative_to(
            run_dir
        ).as_posix()
        if checkpoint_relative not in artifact_entries:
            raise StudyConfigurationError(
                "Artifact manifest omits the selected checkpoint manifest"
            )
        model_path = Path(str(checkpoint.get("merged_model_path", ""))).expanduser()
        weight_files = (
            list(model_path.glob("*.safetensors"))
            + list(model_path.glob("*.bin"))
        ) if model_path.is_dir() else []
        if not (model_path / "config.json").is_file() or not weight_files:
            raise StudyConfigurationError(f"Merged model does not exist: {model_path}")
        checkpoint_sha = artifact_fingerprint(
            str(model_path), allow_missing=False
        )["sha256"]
        if checkpoint.get("merged_model_sha256") != checkpoint_sha:
            raise StudyConfigurationError(
                "Checkpoint manifest model hash does not match the model"
            )
        fixed_config = resolved.get("fixed_test", {})
        expected_comparison_count = int(
            bool(fixed_config.get("evaluate_baseline", True))
        )
        if bool(
            fixed_config.get("evaluate_after_each_training_cycle", True)
        ):
            expected_comparison_count += len(checkpoint_manifests)
        if expected_comparison_count < 1:
            raise StudyConfigurationError(
                "Training run is configured without a fixed evaluation"
            )
        if len(comparisons) != expected_comparison_count:
            raise StudyConfigurationError(
                "Fixed-test evaluation count is incomplete: "
                f"actual={len(comparisons)}, "
                f"expected={expected_comparison_count}"
            )
    dataset = snapshot
    evaluation_sha = dataset.get("sha256")
    evaluation_id = dataset.get("evaluation_set_id")
    evaluation_version = dataset.get(
        "evaluation_version",
        dataset.get("version", dataset.get("dataset_version")),
    )
    if not evaluation_sha or not evaluation_id or not evaluation_version:
        raise StudyConfigurationError(
            "Fixed-test snapshot lacks evaluation-set identity"
        )
    completion = {
        "run_dir": str(run_dir),
        "run_manifest_path": str(expected_paths["run_manifest"]),
        "summary_path": str(expected_paths["summary"]),
        "artifact_manifest_path": str(expected_paths["artifact_manifest"]),
        "artifact_manifest_sha256": file_sha256(expected_paths["artifact_manifest"]),
        "checkpoint_sha256": checkpoint_sha,
        "evaluation_set_id": evaluation_id,
        "evaluation_set_version": evaluation_version,
        "evaluation_set_sha256": evaluation_sha,
        "question_count": len(expected_ids),
    }
    if record.status == "completed":
        persisted_fields = (
            "artifact_manifest_sha256",
            "checkpoint_sha256",
            "evaluation_set_id",
            "evaluation_set_version",
            "evaluation_set_sha256",
        )
        for field in persisted_fields:
            stored = getattr(record, field)
            if not stored or stored != completion[field]:
                raise StudyConfigurationError(
                    f"Completed registry binding mismatch for {field}: "
                    f"{stored!r} != {completion[field]!r}"
                )
    return completion


def _slug(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    return normalized.strip("-").lower() or "unnamed"


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise StudyConfigurationError(f"Study suite must be a mapping: {path}")
    return payload


def _model_entry(value: Any) -> dict[str, str]:
    if isinstance(value, str):
        return {"name": value, "path": value}
    if isinstance(value, Mapping) and value.get("name") and value.get("path"):
        return {"name": str(value["name"]), "path": str(value["path"])}
    raise StudyConfigurationError(
        "Each model must be a string or a mapping with name and path."
    )


def _protocol_entry(value: Any) -> dict[str, Any]:
    if isinstance(value, str):
        return {"name": value}
    if isinstance(value, Mapping) and value.get("name"):
        return dict(value)
    raise StudyConfigurationError(
        "Each protocol must be a string or a mapping with a name."
    )


def _override(path: str, value: Any) -> str:
    encoded = yaml.safe_dump(
        value,
        default_flow_style=True,
        width=10_000,
    ).strip()
    return f"{path}={encoded}"


def _remove_keys(payload: dict[str, Any], dotted_paths: Iterable[str]) -> None:
    for dotted in dotted_paths:
        parts = dotted.split(".")
        cursor: Any = payload
        for part in parts[:-1]:
            if not isinstance(cursor, dict):
                cursor = None
                break
            cursor = cursor.get(part)
        if isinstance(cursor, dict):
            cursor.pop(parts[-1], None)


class StudyRunner:
    def __init__(
        self,
        suite_path: str | Path,
        *,
        project_root: str | Path,
        executor: Callable[[list[str], Path], int] | None = None,
        allow_dirty_worktree: bool = False,
    ):
        self.project_root = Path(project_root).resolve()
        self.suite_path = Path(suite_path)
        if not self.suite_path.is_absolute():
            self.suite_path = (self.project_root / self.suite_path).resolve()
        raw = _load_yaml(self.suite_path)
        self.suite = raw.get("study_suite", raw)
        if not isinstance(self.suite, dict):
            raise StudyConfigurationError("study_suite must be a mapping")
        self.name = str(self.suite["name"])
        output_root = Path(str(self.suite["output_root"])).expanduser()
        self.study_root = (output_root / self.name).resolve()
        self.index_path = self.study_root / "experiment_index.json"
        self.executor = executor or self._subprocess_executor
        self.allow_dirty_worktree = bool(allow_dirty_worktree)
        self._artifact_cache: dict[str, dict[str, Any]] = {}

    @staticmethod
    def _subprocess_executor(command: list[str], cwd: Path) -> int:
        return subprocess.run(command, cwd=cwd, check=False).returncode

    def _method_config(self, method: str) -> Path:
        custom = self.suite.get("method_configs", {})
        raw = custom.get(method, METHOD_CONFIGS.get(method))
        if not raw:
            raise StudyConfigurationError(f"Unknown study method: {method}")
        path = Path(str(raw))
        if not path.is_absolute():
            path = self.project_root / path
        if not path.is_file():
            raise StudyConfigurationError(
                f"Method config does not exist for {method}: {path}"
            )
        return path.resolve()

    def _environment_path(self) -> Path | None:
        raw = self.suite.get("environment")
        if not raw:
            return None
        path = Path(str(raw))
        if not path.is_absolute():
            path = self.project_root / path
        return path.resolve()

    def _common_overrides(self) -> list[str]:
        values = self.suite.get("common_overrides", [])
        if not isinstance(values, list):
            raise StudyConfigurationError("common_overrides must be a list")
        return [str(value) for value in values]

    def _artifact(self, value: str, allow_missing: bool) -> dict[str, Any]:
        key = f"{allow_missing}:{value}"
        if key not in self._artifact_cache:
            self._artifact_cache[key] = artifact_fingerprint(
                value,
                project_root=self.project_root,
                allow_missing=allow_missing,
            )
        return copy.deepcopy(self._artifact_cache[key])

    def build_plan(self) -> list[ExperimentRecord]:
        repository = git_state(self.project_root)
        if repository["dirty"] and not bool(
            self.allow_dirty_worktree
            or self.suite.get("allow_dirty_worktree", False)
        ):
            raise StudyConfigurationError(
                "Refusing to plan a scientific study from a dirty worktree. "
                "Commit the baseline first, or set allow_dirty_worktree: true "
                "only for development/smoke diagnostics."
            )
        current_commit = str(repository["commit"])
        methods = []
        for raw in self.suite["methods"]:
            name = METHOD_ALIASES.get(str(raw), str(raw))
            if name not in methods:
                methods.append(name)
        seeds = [int(value) for value in self.suite["seeds"]]
        models = [_model_entry(value) for value in self.suite["models"]]
        budgets = list(self.suite["budgets"])
        protocols = [
            _protocol_entry(value)
            for value in self.suite.get(
                "protocols",
                [{"name": "question_matched"}],
            )
        ]
        if not all((methods, seeds, models, budgets)):
            raise StudyConfigurationError(
                "methods, seeds, models, and budgets must be non-empty"
            )

        environment = self._environment_path()
        records = []
        resolved_by_id: dict[str, dict[str, Any]] = {}
        for model in models:
            for method in methods:
                config_path = self._method_config(method)
                base_config, _ = load_resolved_config(
                    config_path,
                    environment,
                    temporary_overrides=self._common_overrides(),
                    validate_paths=False,
                )
                for seed in seeds:
                    for raw_budget in budgets:
                        for protocol in protocols:
                            protocol_name = str(protocol["name"])
                            budget = parse_budget(
                                raw_budget,
                                int(base_config["experiment"]["num_iterations"]),
                                int(base_config["experiment"]["max_cycles"]),
                            )
                            experiment_dir = (
                                self.study_root
                                / _slug(model["name"])
                                / protocol_name
                                / method
                                / f"seed_{seed}"
                                / f"budget_{budget.total_questions}"
                            )
                            overrides = [
                            *self._common_overrides(),
                            _override("experiment.seed", seed),
                            *budget.overrides(),
                            _override("budget.protocol", protocol_name),
                            _override(
                                "models.test_taker.model_path",
                                model["path"],
                            ),
                            ]
                            if protocol_name == "data_matched":
                                overrides.append(
                                    _override(
                                        "budget.data_matched_target_samples",
                                        int(protocol["target_training_samples"]),
                                    )
                                )
                            if protocol_name == "generation_token_matched":
                                overrides.append(
                                    _override(
                                        "budget.max_generation_tokens",
                                        int(protocol["max_generation_tokens"]),
                                    )
                                )
                                if protocol.get("max_total_api_calls") is not None:
                                    overrides.append(
                                        _override(
                                            "budget.max_total_api_calls",
                                            int(protocol["max_total_api_calls"]),
                                        )
                                    )
                            for field in RUNTIME_PATH_FIELDS:
                                overrides.append(
                                _override(
                                    f"paths.{field}",
                                    str(experiment_dir / field),
                                )
                                )
                            overrides.append(
                            _override(
                                "paths.outfile_prefix",
                                str(experiment_dir / "legacy" / "output"),
                            )
                            )
                            config, provenance = load_resolved_config(
                            config_path,
                            environment,
                            temporary_overrides=overrides,
                            validate_paths=False,
                            )
                            identity = {
                            "suite": self.name,
                            "method": method,
                            "variant": str(config["study"]["variant"]),
                            "budget_protocol": protocol_name,
                            "seed": seed,
                            "model": model["name"],
                            "budget": budget.total_questions,
                            "config_hash": provenance["config_hash"],
                            "git_commit": current_commit,
                            }
                            study_id = (
                            f"{_slug(self.name)}-{_slug(model['name'])}-"
                            f"{protocol_name}-{method}-s{seed}-"
                            f"b{budget.total_questions}-"
                            f"{canonical_sha256(identity)[:10]}"
                            )
                            command = [
                            sys.executable,
                            "-B",
                            str(self.project_root / "math_autobencher.py"),
                            "--use_helm",
                            "no",
                            "--config",
                            str(config_path),
                            ]
                            if environment:
                                command.extend(["--environment", str(environment)])
                            command.extend(
                            [
                                "--run_id",
                                study_id,
                                "--resume",
                                "true",
                                "--override",
                                *overrides,
                            ]
                            )
                            allow_missing = bool(
                            self.suite.get("allow_missing_artifacts", False)
                            )
                            fixed_path = resolve_project_path(
                            self.project_root,
                            str(config["fixed_test"]["dataset_path"]),
                            )
                            fingerprints = {
                            "base_model": self._artifact(
                                str(config["models"]["test_taker"]["model_path"]),
                                allow_missing,
                            ),
                            "fixed_test": self._artifact(
                                str(fixed_path),
                                allow_missing,
                            ),
                            "evaluation_registry": self._artifact(
                                str(
                                    resolve_project_path(
                                        self.project_root,
                                        str(
                                            config["evaluation_sets"][
                                                "registry_path"
                                            ]
                                        ),
                                    )
                                ),
                                allow_missing,
                            ),
                            "generation_guidance": (
                                self._artifact(
                                    str(
                                        config["generation_guidance"][
                                            "dataset_path"
                                        ]
                                    ),
                                    allow_missing,
                                )
                                if bool(
                                    config["generation_guidance"]["enabled"]
                                )
                                else {
                                    "kind": "disabled",
                                    "sha256": canonical_sha256(
                                        {"enabled": False}
                                    ),
                                }
                            ),
                            "prompt_bundle": prompt_bundle_snapshot(
                                config,
                                self.project_root,
                            ),
                            "training_config_sha256": canonical_sha256(
                                thaw_config(config["finetune"])
                            ),
                            }
                            record = ExperimentRecord(
                            study_id=study_id,
                            method=method,
                            variant=str(config["study"]["variant"]),
                            seed=seed,
                            model=model["name"],
                            budget=budget.total_questions,
                            config_hash=str(provenance["config_hash"]),
                            git_commit=current_commit,
                            budget_protocol=protocol_name,
                            experiment_dir=str(experiment_dir),
                            command=command,
                            fingerprints=fingerprints,
                            run_dir=str(
                                run_dir_for_id(
                                    config["paths"]["output_root"],
                                    study_id,
                                )
                            ),
                            )
                            record.run_manifest_path = str(
                                Path(record.run_dir) / "run_manifest.json"
                            )
                            record.summary_path = str(
                                Path(record.run_dir) / "experiment_summary.json"
                            )
                            record.artifact_manifest_path = str(
                                Path(record.run_dir) / "artifact_manifest.json"
                            )
                            records.append(record)
                            resolved_by_id[study_id] = config
        self._validate_isolation(records)
        self._validate_fairness(records, resolved_by_id)
        return records

    @staticmethod
    def _validate_isolation(records: list[ExperimentRecord]) -> None:
        directories = [
            os.path.normcase(os.path.abspath(record.experiment_dir))
            for record in records
        ]
        if len(directories) != len(set(directories)):
            raise StudyConfigurationError(
                "Two experiments resolve to the same experiment directory."
            )
        for index, left in enumerate(directories):
            for right in directories[index + 1 :]:
                try:
                    common = os.path.commonpath([left, right])
                except ValueError:
                    continue
                if common in {left, right}:
                    raise StudyConfigurationError(
                        "Experiment directories must not contain one another: "
                        f"{left!r}, {right!r}"
                    )

    def _validate_fairness(
        self,
        records: list[ExperimentRecord],
        configs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        groups: dict[
            tuple[str, str, int, int],
            list[ExperimentRecord],
        ] = defaultdict(list)
        for record in records:
            groups[
                (
                    record.model,
                    record.budget_protocol,
                    record.seed,
                    record.budget,
                )
            ].append(record)
        for key, group in groups.items():
            model_hashes = {
                item.fingerprints["base_model"]["sha256"] for item in group
            }
            fixed_hashes = {
                item.fingerprints["fixed_test"]["sha256"] for item in group
            }
            registry_hashes = {
                item.fingerprints["evaluation_registry"]["sha256"]
                for item in group
            }
            prompt_hashes = {
                item.fingerprints["prompt_bundle"]["combined_sha256"]
                for item in group
            }
            guidance_hashes = {
                item.fingerprints["generation_guidance"]["sha256"]
                for item in group
            }
            if len(model_hashes) != 1:
                raise StudyConfigurationError(
                    f"Unfair base model fingerprints in matrix cell {key}."
                )
            if len(fixed_hashes) != 1:
                raise StudyConfigurationError(
                    f"Unfair fixed-test fingerprints in matrix cell {key}."
                )
            if len(registry_hashes) != 1:
                raise StudyConfigurationError(
                    "Unfair evaluation-registry fingerprints in matrix "
                    f"cell {key}."
                )
            if len(prompt_hashes) != 1:
                raise StudyConfigurationError(
                    f"Unfair prompt fingerprints in matrix cell {key}."
                )
            if len(guidance_hashes) != 1:
                raise StudyConfigurationError(
                    "Unfair generation-guidance fingerprints in matrix "
                    f"cell {key}."
                )

            training_methods = [item for item in group if item.method != "base"]
            requested_fields = set(
                self.suite.get("method_specific_fields", [])
            )
            unsupported = requested_fields - ALLOWED_METHOD_SPECIFIC_FIELDS
            if unsupported:
                raise StudyConfigurationError(
                    "Unsupported method_specific_fields; scientific suites "
                    "may only vary explicitly audited strategy fields: "
                    f"{sorted(unsupported)}"
                )
            normalized = {}
            for item in training_methods:
                payload = thaw_config(configs[item.study_id])
                _remove_keys(payload, ("study", "paths"))
                _remove_keys(payload, requested_fields)
                normalized[item.method] = canonical_sha256(payload)
            if len(set(normalized.values())) > 1:
                raise StudyConfigurationError(
                    "Unfair non-strategy configuration change in matrix cell "
                    f"{key}: {normalized}"
                )

    def _registry_payload(
        self,
        records: list[ExperimentRecord],
    ) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "suite_name": self.name,
            "suite_path": str(self.suite_path),
            "suite_sha256": canonical_sha256(self.suite),
            "git_commit": git_commit(self.project_root),
            "comparison_pairs": list(
                self.suite.get("comparison_pairs", [])
            ),
            "updated_at": utc_now(),
            "experiments": [record.to_dict() for record in records],
        }

    def initialize_registry(
        self,
        plan: list[ExperimentRecord],
        *,
        resume: bool,
    ) -> list[ExperimentRecord]:
        if not self.index_path.exists():
            atomic_json(self._registry_payload(plan), self.index_path)
            return plan
        with self.index_path.open("r", encoding="utf-8") as handle:
            existing_payload = json.load(handle)
        validate_registry(existing_payload)
        existing = {
            item["study_id"]: ExperimentRecord.from_dict(item)
            for item in existing_payload["experiments"]
        }
        planned_ids = [item.study_id for item in plan]
        if set(existing) != set(planned_ids):
            raise StudyConfigurationError(
                "Existing experiment index does not match the current matrix. "
                "Use a new study_suite.name or restore the original suite."
            )
        if not resume and any(
            item.status != "pending" for item in existing.values()
        ):
            raise StudyConfigurationError(
                "The study already has progress; pass --resume to continue."
            )
        merged = []
        for planned in plan:
            old = existing[planned.study_id]
            if not old.run_dir:
                if old.status != "pending" or any(
                    Path(old.experiment_dir).rglob("run_manifest.json")
                ):
                    raise StudyConfigurationError(
                        "Legacy registry has no deterministic run_dir for "
                        f"{old.study_id}; use a new suite name rather than "
                        "guessing among test_N attempts."
                    )
                old.run_dir = planned.run_dir
                old.run_manifest_path = planned.run_manifest_path
                old.summary_path = planned.summary_path
                old.artifact_manifest_path = planned.artifact_manifest_path
            if (
                old.config_hash != planned.config_hash
                or old.git_commit != planned.git_commit
                or old.command != planned.command
                or old.fingerprints != planned.fingerprints
                or old.run_dir != planned.run_dir
            ):
                raise StudyConfigurationError(
                    "Code/config/artifact fingerprints changed for existing "
                    f"experiment {planned.study_id}; use a new suite name."
                )
            if old.status == "running":
                old.status = "partial"
                old.error = "Recovered stale running state"
            merged.append(old)
        atomic_json(self._registry_payload(merged), self.index_path)
        return merged

    def _save(self, records: list[ExperimentRecord]) -> None:
        atomic_json(self._registry_payload(records), self.index_path)

    def run(
        self,
        *,
        resume: bool = False,
        dry_run: bool = False,
        methods: set[str] | None = None,
        study_ids: set[str] | None = None,
        max_experiments: int | None = None,
    ) -> list[ExperimentRecord]:
        plan = self.build_plan()
        records = self.initialize_registry(plan, resume=resume)
        if dry_run:
            return records
        selected = 0
        for record in records:
            if record.status == "completed":
                continue
            if methods and record.method not in methods:
                continue
            if study_ids and record.study_id not in study_ids:
                continue
            if max_experiments is not None and selected >= max_experiments:
                break
            selected += 1
            Path(record.experiment_dir).mkdir(parents=True, exist_ok=True)
            record.status = "running"
            record.started_at = utc_now()
            record.error = None
            self._save(records)
            try:
                return_code = int(
                    self.executor(record.command, self.project_root)
                )
                record.return_code = return_code
                record.completed_at = utc_now()
                if return_code == 0:
                    try:
                        completion = validate_experiment_completion(record)
                    except StudyConfigurationError as error:
                        record.status = "partial"
                        record.error = f"Completion validation failed: {error}"
                    else:
                        for key, value in completion.items():
                            if hasattr(record, key):
                                setattr(record, key, value)
                        record.status = "completed"
                else:
                    has_artifacts = any(
                        Path(record.experiment_dir).rglob("*")
                    )
                    record.status = "partial" if has_artifacts else "failed"
                    record.error = f"Process exited with code {return_code}"
            except (Exception, KeyboardInterrupt) as exc:
                record.completed_at = utc_now()
                record.status = "partial"
                record.error = f"{type(exc).__name__}: {exc}"
                self._save(records)
                if isinstance(exc, KeyboardInterrupt):
                    raise
            self._save(records)
        return records
