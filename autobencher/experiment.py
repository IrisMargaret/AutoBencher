"""Research manifests, atomic exports, structured logging, and progress."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from autobencher.config import ProjectConfig, thaw_config
from autobencher.budget_ledger import BudgetLedger, set_active_ledger
from autobencher.fingerprints import (
    artifact_fingerprint,
    canonical_sha256,
    file_sha256,
    prompt_bundle_snapshot,
)


ARTIFACT_LAYOUT_VERSION = "research_run_v2"
HASHED_ARTIFACT_SUFFIXES = frozenset(
    {".csv", ".json", ".jsonl", ".log", ".md", ".txt", ".yaml", ".yml"}
)


def utc_now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def git_commit(cwd: str | Path | None = None) -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return "unknown"
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def atomic_json(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_yaml(data: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        yaml.safe_dump(
            thaw_config(data),
            handle,
            allow_unicode=True,
            sort_keys=False,
            default_flow_style=False,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_text(text: str, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(str(text))
        if not str(text).endswith("\n"):
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def allocate_test_run_dir(output_root: str | Path) -> Path:
    """Atomically allocate test_<max+1> beneath the configured output root."""
    root = Path(output_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(r"^test_(\d+)$")
    numbers = [
        int(match.group(1))
        for path in root.iterdir()
        if path.is_dir()
        for match in [pattern.fullmatch(path.name)]
        if match
    ]
    candidate_number = max(numbers, default=0) + 1
    while True:
        candidate = root / f"test_{candidate_number}"
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate_number += 1


def run_dir_for_id(output_root: str | Path, run_id: str) -> Path:
    """Map one run identity to exactly one stable directory."""
    root = Path(output_root).expanduser().resolve()
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", str(run_id)).strip("-.")
    if not normalized:
        raise ValueError("run_id must contain at least one safe character")
    digest = hashlib.sha256(str(run_id).encode("utf-8")).hexdigest()[:12]
    return root / f"run_{normalized[:96]}_{digest}"


def _artifact_segment(value: Any, field: str) -> str:
    normalized = str(value).strip()
    if not normalized or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,95}", normalized):
        raise ValueError(
            f"{field} must be a safe artifact name containing only letters, "
            "numbers, dot, underscore, and hyphen"
        )
    return normalized


def build_artifact_inventory(run_dir: str | Path) -> dict[str, Any]:
    """Create a deterministic index of finalized run outputs.

    Every file is content-addressed. This is intentionally performed only when
    a run is finalized; complete integrity is more important than avoiding one
    sequential read of model payloads at publication time.
    """
    root = Path(run_dir).resolve()
    files = []
    for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
        if not path.is_file() or path.name == "artifact_manifest.json":
            continue
        relative = path.relative_to(root).as_posix()
        if path.is_symlink() and not path.resolve().is_relative_to(root):
            raise RuntimeError(f"Artifact symlink escapes run directory: {relative}")
        if path.name.endswith(".tmp"):
            raise RuntimeError(f"Uncommitted temporary artifact remains: {relative}")
        files.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "sha256": file_sha256(path),
                "kind": (
                    "research_artifact"
                    if path.suffix.lower() in HASHED_ARTIFACT_SUFFIXES
                    else "binary_payload"
                ),
            }
        )
    return {
        "layout_version": ARTIFACT_LAYOUT_VERSION,
        "hash_policy": {
            "algorithm": "sha256",
            "coverage": "all_regular_files",
            "research_artifact_suffixes": sorted(HASHED_ARTIFACT_SUFFIXES),
            "temporary_files_allowed": False,
        },
        "files": files,
    }


def environment_snapshot() -> dict[str, Any]:
    packages = {}
    for name in (
        "torch",
        "transformers",
        "peft",
        "accelerate",
        "trl",
        "bitsandbytes",
        "datasets",
        "sympy",
        "math-verify",
        "sentence-transformers",
        "datasketch",
        "outlines",
        "guidance",
        "wandb",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    cuda = {
        "available": False,
        "runtime": None,
        "device_count": 0,
        "devices": [],
    }
    try:
        import torch

        cuda.update(
            {
                "available": bool(torch.cuda.is_available()),
                "runtime": torch.version.cuda,
                "device_count": (
                    torch.cuda.device_count() if torch.cuda.is_available() else 0
                ),
                "devices": (
                    [
                        torch.cuda.get_device_name(index)
                        for index in range(torch.cuda.device_count())
                    ]
                    if torch.cuda.is_available()
                    else []
                ),
            }
        )
    except (ImportError, RuntimeError):
        pass
    return {
        "created_at": utc_now(),
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "packages": packages,
        "cuda": cuda,
        "image_name": os.getenv("CONTAINER_IMAGE_NAME"),
        "image_tag": os.getenv("CONTAINER_IMAGE_TAG"),
        "image_digest": os.getenv("CONTAINER_IMAGE_DIGEST"),
        "driver_version": os.getenv("NVIDIA_DRIVER_VERSION"),
    }


def study_manifest_snapshot(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return declared study settings alongside the resolved runtime policy."""
    # Import lazily to keep experiment primitives independent of policy module
    # import order and to avoid a future coverage/experiment import cycle.
    from autobencher.policies import policy_runtime_descriptor

    config_snapshot = thaw_config(config["study"])
    runtime = thaw_config(policy_runtime_descriptor(config))
    runtime.setdefault("seed", int(config["experiment"]["seed"]))
    runtime.setdefault(
        "question_budget",
        int(config["experiment"]["questions_per_iteration"]),
    )
    runtime.setdefault(
        "question_budget_per_iteration",
        int(config["experiment"]["questions_per_iteration"]),
    )
    runtime.setdefault(
        "total_question_budget",
        int(config["experiment"]["questions_per_iteration"])
        * int(config["experiment"]["num_iterations"])
        * int(config["experiment"]["max_cycles"]),
    )
    return {
        "config_snapshot": config_snapshot,
        **runtime,
    }


class ExperimentLogger:
    def __init__(
        self,
        run_id: str,
        log_dir: str | Path,
        config: Mapping[str, Any],
    ):
        self.run_id = run_id
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.text_path = self.log_dir / "run.log"
        self.events_path = self.log_dir / "events.jsonl"
        self.config = config
        self._lock = threading.Lock()
        self._last_message = None

    def event(
        self,
        level: str,
        stage: str,
        event: str,
        message: str = "",
        cycle: int | None = None,
        iteration: int | None = None,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        timestamp = utc_now()
        payload = {
            "timestamp": timestamp,
            "level": level.upper(),
            "run_id": self.run_id,
            "cycle": cycle,
            "iteration": iteration,
            "stage": stage,
            "event": event,
            "message": message,
            "metrics": dict(metrics or {}),
        }
        cycle_text = f"Cycle {cycle}" if cycle is not None else "Cycle -"
        iteration_text = (
            f"Iter {iteration}" if iteration is not None else "Iter -"
        )
        line = (
            f"[{timestamp}][{level.upper()}][run_id={self.run_id}]"
            f"[{cycle_text}][{iteration_text}][{stage}] {message}"
        ).rstrip()
        with self._lock:
            if self.config["logging"]["file_enabled"]:
                with self.text_path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
            if self.config["logging"]["jsonl_enabled"]:
                with self.events_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            suppress = bool(
                self.config["logging"]["suppress_duplicate_messages"]
            )
            if self.config["logging"]["console_enabled"] and not (
                suppress and line == self._last_message
            ):
                print(line)
            self._last_message = line


class ProgressManager:
    def __init__(self, config: Mapping[str, Any]):
        self.config = config
        self._active = None
        if config["logging"]["disable_library_progress"]:
            os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] = "1"
            os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "1"
            try:
                from datasets.utils.logging import disable_progress_bar

                disable_progress_bar()
            except ImportError:
                pass
            try:
                from transformers.utils.logging import disable_progress_bar

                disable_progress_bar()
            except ImportError:
                pass

    @contextlib.contextmanager
    def stage(
        self,
        name: str,
        total: int,
        cycle: int | None = None,
        iteration: int | None = None,
    ):
        if self._active is not None:
            raise RuntimeError(
                f"progress stage {self._active} is still active; cannot start {name}"
            )
        enabled = bool(self.config["logging"]["progress_enabled"]) and bool(
            getattr(sys.stderr, "isatty", lambda: False)()
        )
        progress = None
        self._active = name
        try:
            if enabled:
                from tqdm.auto import tqdm

                description = " ".join(
                    part
                    for part in (
                        f"C{cycle}" if cycle is not None else "",
                        f"I{iteration}" if iteration is not None else "",
                        name,
                    )
                    if part
                )
                progress = tqdm(
                    total=total,
                    desc=description,
                    leave=bool(self.config["logging"]["progress_leave"]),
                    dynamic_ncols=bool(
                        self.config["logging"]["progress_dynamic_ncols"]
                    ),
                    position=0,
                    mininterval=0.2,
                    bar_format=(
                        "{desc}: {percentage:6.2f}%|{bar}| "
                        "{n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
                    ),
                )
            yield progress or _NullProgress()
        finally:
            if progress is not None:
                progress.close()
            self._active = None


class _NullProgress:
    def update(self, amount: int = 1) -> None:
        del amount

    def set_postfix(self, **values: Any) -> None:
        del values


class ResearchRun:
    def __init__(
        self,
        config: Mapping[str, Any],
        provenance: Mapping[str, Any],
        run_id: str,
        project_root: str | Path,
        legacy_output_root: str | Path | None = None,
        resume: bool = False,
    ):
        self.config = (
            config
            if isinstance(config, ProjectConfig)
            else ProjectConfig(config)
        )
        self.provenance = dict(provenance)
        self.run_id = run_id
        self.project_root = Path(project_root).resolve()
        configured_root = Path(str(config["paths"]["output_root"])).expanduser()
        self.output_root = (
            Path(legacy_output_root).resolve()
            if legacy_output_root is not None
            else configured_root.resolve()
        )
        self.run_dir = run_dir_for_id(self.output_root, run_id)
        self.resumed = self.run_dir.is_dir()
        if self.resumed and not resume:
            raise RuntimeError(
                f"Run directory already exists for run_id={run_id!r}; "
                "enable resume or choose a different run_id"
            )
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.cycle_root = self.run_dir / "cycle"
        self.cycle_root.mkdir(parents=True, exist_ok=True)
        log_dir = self.run_dir / "logs"
        self.logger = ExperimentLogger(run_id, log_dir, config)
        self.progress = ProgressManager(config)
        self.git_commit = git_commit(self.project_root)
        self.config_hash = str(provenance["config_hash"])
        self.prompt_bundle = prompt_bundle_snapshot(
            self.config,
            self.project_root,
        )
        self.base_model_fingerprint = artifact_fingerprint(
            str(self.config["models"]["test_taker"]["model_path"]),
            project_root=self.project_root,
            allow_missing=False,
        )
        guidance_config = self.config["generation_guidance"]
        if bool(guidance_config["enabled"]):
            self.generation_guidance_fingerprint = artifact_fingerprint(
                str(guidance_config["dataset_path"]),
                project_root=self.project_root,
                allow_missing=False,
            )
        else:
            self.generation_guidance_fingerprint = {
                "kind": "disabled",
                "sha256": canonical_sha256({"enabled": False}),
            }
        if self.resumed:
            manifest_path = self.run_dir / "run_manifest.json"
            if not manifest_path.is_file():
                raise RuntimeError(
                    "Cannot resume: deterministic run directory has no "
                    "run_manifest.json"
                )
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            expected = {
                "run_id": self.run_id,
                "config_hash": self.config_hash,
                "git_commit": self.git_commit,
                "prompt_hash": self.prompt_bundle["combined_sha256"],
                "base_model_sha256": self.base_model_fingerprint["sha256"],
                "generation_guidance_sha256": (
                    self.generation_guidance_fingerprint["sha256"]
                ),
            }
            mismatches = {
                key: {"stored": existing.get(key), "current": value}
                for key, value in expected.items()
                if existing.get(key) != value
            }
            if mismatches:
                raise RuntimeError(
                    "Resume identity validation failed before opening mutable "
                    "state: " + json.dumps(mismatches, ensure_ascii=False)
                )
            if existing.get("status") == "completed":
                raise RuntimeError(
                    "Cannot resume an already completed run; use its bound "
                    "artifacts or choose a new run_id"
                )
        self.budget_ledger = BudgetLedger(
            self.run_dir / "budget_ledger.json",
            thaw_config(self.config["budget"]),
            self.metadata(),
            resume=self.resumed,
        )
        set_active_ledger(self.budget_ledger)

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": str(self.config["export"]["schema_version"]),
            "run_id": self.run_id,
            "created_at": utc_now(),
            "git_commit": self.git_commit,
            "config_hash": self.config_hash,
        }

    def initialize(self, cli_args: Mapping[str, Any]) -> None:
        resolved_payload = thaw_config(self.config)
        study_snapshot = study_manifest_snapshot(self.config)
        difficulty_snapshot = thaw_config(self.config["difficulty"])
        calibration_path = difficulty_snapshot.get("calibration_artifact")
        if calibration_path and Path(str(calibration_path)).is_file():
            difficulty_snapshot["calibration_artifact_actual_sha256"] = (
                file_sha256(calibration_path)
            )
        prompt_bundle = self.prompt_bundle
        base_model = self.base_model_fingerprint
        generation_guidance = self.generation_guidance_fingerprint
        manifest_path = self.run_dir / "run_manifest.json"
        if self.resumed:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            atomic_json(
                {
                    **existing,
                    "status": "running",
                    "resumed_at": utc_now(),
                    "resume_count": int(existing.get("resume_count", 0)) + 1,
                    "cli_args": dict(cli_args),
                },
                manifest_path,
            )
            self.logger.event(
                "INFO",
                "Startup",
                "run_resumed",
                f"run_dir={self.run_dir}",
            )
            return
        atomic_json(resolved_payload, self.run_dir / "resolved_config.json")
        atomic_yaml(resolved_payload, self.run_dir / "resolved_config.yaml")
        atomic_json(
            {
                "schema_version": self.provenance["schema_version"],
                "config_hash": self.config_hash,
                "sources": self.provenance.get("sources", {}),
                "field_sources": self.provenance.get("field_sources", {}),
                "sensitive_environment": self.provenance.get(
                    "sensitive_environment",
                    {},
                ),
                "migrations": self.provenance.get("migrations", []),
            },
            self.run_dir / "config_sources.json",
        )
        atomic_json(
            {
                "schema_version": self.provenance["schema_version"],
                "config_hash": self.config_hash,
                **self.provenance.get("validation", {}),
            },
            self.run_dir / "config_validation.json",
        )
        atomic_text(self.config_hash, self.run_dir / "config_hash.txt")
        environment = {**self.metadata(), **environment_snapshot()}
        atomic_json(environment, self.run_dir / "environment.json")
        atomic_json(
            {
                **self.metadata(),
                "status": "running",
                "resume_count": 0,
                "base_model": base_model,
                "base_model_sha256": base_model["sha256"],
                "generation_guidance": generation_guidance,
                "generation_guidance_sha256": generation_guidance["sha256"],
                "cli_args": dict(cli_args),
                "study": study_snapshot,
                "study_config_snapshot": study_snapshot["config_snapshot"],
                "policy_name": study_snapshot.get("policy_name"),
                "policy_version": study_snapshot.get("policy_version"),
                "variant": study_snapshot.get("variant"),
                "component_state": study_snapshot.get("component_state", {}),
                "component_evidence": study_snapshot.get(
                    "component_evidence",
                    {},
                ),
                "adaptive_history": {
                    "mode": self.config["adaptive_sampling"]["history_mode"],
                    "decay_lambda": self.config["adaptive_sampling"][
                        "decay_lambda"
                    ],
                },
                "difficulty": difficulty_snapshot,
                "seed": study_snapshot["seed"],
                "question_budget": study_snapshot["question_budget"],
                "question_budget_per_iteration": study_snapshot[
                    "question_budget_per_iteration"
                ],
                "total_question_budget": study_snapshot[
                    "total_question_budget"
                ],
                "budget_protocol": str(self.config["budget"]["protocol"]),
                "evaluation_provenance": thaw_config(
                    self.config["evaluation_provenance"]
                ),
                "prompt_version": "content-addressed-v1",
                "prompt_hash": prompt_bundle["combined_sha256"],
                "prompt_bundle": prompt_bundle,
            },
            self.run_dir / "run_manifest.json",
        )
        atomic_json(
            {
                **self.metadata(),
                "taxonomy": self.config["taxonomy"],
            },
            self.run_dir / "taxonomy_snapshot.json",
        )

    def iteration_dir(self, cycle: int, iteration: int) -> Path:
        if int(cycle) < 1 or int(iteration) < 1:
            raise ValueError("cycle and iteration identifiers must be positive")
        path = self.cycle_root / f"cycle_{cycle}" / f"iter_{iteration}"
        path.mkdir(parents=True, exist_ok=True)
        return path

    def export_iteration(
        self,
        cycle: int,
        iteration: int,
        payloads: Mapping[str, Any],
    ) -> None:
        target = self.iteration_dir(cycle, iteration)
        for name, payload in payloads.items():
            artifact_name = _artifact_segment(name, "iteration artifact name")
            atomic_json(
                {**self.metadata(), "cycle_id": cycle, "iteration_id": iteration, "data": payload},
                target / f"{artifact_name}.json",
            )

    def save_cycle_artifact(
        self,
        cycle: int,
        section: str,
        name: str,
        payload: Any,
    ) -> Path:
        """Persist one metadata-enveloped artifact within a cycle section."""
        if int(cycle) < 1:
            raise ValueError("cycle identifier must be positive")
        section_name = _artifact_segment(section, "cycle artifact section")
        artifact_name = _artifact_segment(name, "cycle artifact name")
        target = (
            self.cycle_root
            / f"cycle_{cycle}"
            / section_name
            / f"{artifact_name}.json"
        )
        atomic_json(
            {
                **self.metadata(),
                "cycle_id": cycle,
                "section": section,
                "data": payload,
            },
            target,
        )
        return target

    def finalize(
        self,
        status: str,
        summary: Mapping[str, Any] | None = None,
    ) -> None:
        """Atomically close the run manifest without discarding start metadata."""
        self.budget_ledger.finalize(status)
        manifest_path = self.run_dir / "run_manifest.json"
        existing: dict[str, Any] = {}
        if manifest_path.is_file():
            try:
                existing = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                existing = {}
        atomic_json(
            {
                **existing,
                **self.metadata(),
                "status": status,
                "completed_at": utc_now(),
                "summary": dict(summary or {}),
            },
            manifest_path,
        )
        if status == "completed":
            inventory = build_artifact_inventory(self.run_dir)
            atomic_json(
                {
                    "schema_version": "1.0",
                    "run_id": self.run_id,
                    "config_hash": self.config_hash,
                    "git_commit": self.git_commit,
                    "created_at": utc_now(),
                    **inventory,
                },
                self.run_dir / "artifact_manifest.json",
            )
        set_active_ledger(None)
