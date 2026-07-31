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
from autobencher.experiment import atomic_json, git_commit, utc_now
from autobencher.fingerprints import (
    artifact_fingerprint,
    canonical_sha256,
    prompt_bundle_snapshot,
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
    "full_no_observed_difficulty": (
        "configs/studies/ablations/full_no_observed_difficulty.yaml"
    ),
}

METHOD_ALIASES = {
    "full_no_observed_difficulty_sampling": (
        "full_no_observed_difficulty"
    ),
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
                        budget = parse_budget(
                            raw_budget,
                            int(base_config["experiment"]["num_iterations"]),
                            int(base_config["experiment"]["max_cycles"]),
                        )
                        experiment_dir = (
                            self.study_root
                            / _slug(model["name"])
                            / method
                            / f"seed_{seed}"
                            / f"budget_{budget.total_questions}"
                        )
                        overrides = [
                            *self._common_overrides(),
                            _override("experiment.seed", seed),
                            *budget.overrides(),
                            _override(
                                "models.test_taker.model_path",
                                model["path"],
                            ),
                        ]
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
                            "seed": seed,
                            "model": model["name"],
                            "budget": budget.total_questions,
                            "config_hash": provenance["config_hash"],
                            "git_commit": current_commit,
                        }
                        study_id = (
                            f"{_slug(self.name)}-{_slug(model['name'])}-"
                            f"{method}-s{seed}-b{budget.total_questions}-"
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
                            experiment_dir=str(experiment_dir),
                            command=command,
                            fingerprints=fingerprints,
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

    @staticmethod
    def _validate_fairness(
        records: list[ExperimentRecord],
        configs: Mapping[str, Mapping[str, Any]],
    ) -> None:
        groups: dict[tuple[str, int, int], list[ExperimentRecord]] = defaultdict(list)
        for record in records:
            groups[(record.model, record.seed, record.budget)].append(record)
        for key, group in groups.items():
            model_hashes = {
                item.fingerprints["base_model"]["sha256"] for item in group
            }
            fixed_hashes = {
                item.fingerprints["fixed_test"]["sha256"] for item in group
            }
            prompt_hashes = {
                item.fingerprints["prompt_bundle"]["combined_sha256"]
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
            if len(prompt_hashes) != 1:
                raise StudyConfigurationError(
                    f"Unfair prompt fingerprints in matrix cell {key}."
                )

            training_methods = [item for item in group if item.method != "base"]
            normalized = {}
            for item in training_methods:
                payload = thaw_config(configs[item.study_id])
                _remove_keys(payload, ("study", "paths"))
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
            if (
                old.config_hash != planned.config_hash
                or old.git_commit != planned.git_commit
                or old.command != planned.command
                or old.fingerprints != planned.fingerprints
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
