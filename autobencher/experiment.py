"""Research manifests, atomic exports, structured logging, and progress."""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import re
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from autobencher.config import ProjectConfig, thaw_config


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


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        self.run_dir = allocate_test_run_dir(self.output_root)
        self.cycle_root = self.run_dir / "cycle"
        self.cycle_root.mkdir(parents=True, exist_ok=True)
        log_dir = self.run_dir / "logs"
        self.logger = ExperimentLogger(run_id, log_dir, config)
        self.progress = ProgressManager(config)
        self.git_commit = git_commit(self.project_root)
        self.config_hash = str(provenance["config_hash"])

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
                "cli_args": dict(cli_args),
                "prompt_version": "math_structured_v1",
                "prompt_hash": hashlib.sha256(
                    b"math_structured_v1"
                ).hexdigest(),
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
            atomic_json(
                {**self.metadata(), "cycle_id": cycle, "iteration_id": iteration, "data": payload},
                target / f"{name}.json",
            )

    def save_cycle_artifact(
        self,
        cycle: int,
        section: str,
        name: str,
        payload: Any,
    ) -> Path:
        """Persist one metadata-enveloped artifact within a cycle section."""
        target = self.cycle_root / f"cycle_{cycle}" / section / f"{name}.json"
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
