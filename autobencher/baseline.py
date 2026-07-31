"""Reproducible baseline manifests for ablation studies."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Mapping

from autobencher.config import thaw_config
from autobencher.experiment import environment_snapshot, git_commit, utc_now
from autobencher.fingerprints import (
    artifact_fingerprint,
    canonical_sha256,
    file_sha256,
    prompt_bundle_snapshot,
    resolve_project_path,
)


def git_state(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain=v1"],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return {"commit": "unknown", "dirty": None, "changes": []}
    changes = [
        line.rstrip()
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    return {
        "commit": git_commit(root),
        "dirty": bool(changes) if result.returncode == 0 else None,
        "changes": changes,
    }


def _config_source_fingerprints(
    provenance: Mapping[str, Any],
    project_root: str | Path,
) -> list[dict[str, str]]:
    sources = provenance.get("sources", {})
    candidates = [
        *sources.get("config_layers", []),
        *sources.get("environment_layers", []),
    ]
    records = []
    root = Path(project_root).resolve()
    for raw_path in candidates:
        path = Path(str(raw_path))
        if path.is_file():
            resolved = path.resolve()
            try:
                portable = resolved.relative_to(root).as_posix()
            except ValueError:
                portable = resolved.as_posix()
            records.append(
                {
                    "path": portable,
                    "sha256": file_sha256(path),
                }
            )
    return records


def build_baseline_manifest(
    config: Mapping[str, Any],
    provenance: Mapping[str, Any],
    project_root: str | Path,
    *,
    baseline_id: str = "baseline-ablation-v1",
    allow_missing_artifacts: bool = False,
) -> dict[str, Any]:
    """Build a content-addressed baseline without mutating Git or runtime data."""
    root = Path(project_root).resolve()
    model = artifact_fingerprint(
        str(config["models"]["test_taker"]["model_path"]),
        project_root=root,
        allow_missing=allow_missing_artifacts,
    )
    fixed_path = resolve_project_path(
        root,
        str(config["fixed_test"]["dataset_path"]),
    )
    fixed_test = artifact_fingerprint(
        fixed_path,
        allow_missing=allow_missing_artifacts,
    )
    evaluation_registry = artifact_fingerprint(
        resolve_project_path(
            root,
            str(config["evaluation_sets"]["registry_path"]),
        ),
        allow_missing=allow_missing_artifacts,
    )
    prompts = prompt_bundle_snapshot(config, root)
    config_sources = _config_source_fingerprints(provenance, root)
    resolved_config = thaw_config(config)
    difficulty = thaw_config(config["difficulty"])
    calibration_path = difficulty.get("calibration_artifact")
    difficulty_artifact = (
        artifact_fingerprint(
            str(calibration_path),
            project_root=root,
            allow_missing=allow_missing_artifacts,
        )
        if calibration_path
        else None
    )
    manifest = {
        "schema_version": "1.0",
        "baseline_id": baseline_id,
        "created_at": utc_now(),
        "git": git_state(root),
        "environment": environment_snapshot(),
        "model": {
            "name": str(config["models"]["test_taker"]["model_path"]),
            **model,
        },
        "fixed_test": fixed_test,
        "evaluation_registry": evaluation_registry,
        "config": {
            "resolved_sha256": str(provenance["config_hash"]),
            "source_files": config_sources,
            "source_bundle_sha256": canonical_sha256(config_sources),
        },
        "prompt_bundle": prompts,
        "difficulty": {
            "rubric_version": difficulty["rubric_version"],
            "dimension_weights": difficulty["dimension_weights"],
            "calibration_artifact": difficulty_artifact,
            "sha256": canonical_sha256(difficulty),
        },
    }
    stable_environment = {
        key: value
        for key, value in manifest["environment"].items()
        if key != "created_at"
    }
    manifest["baseline_sha256"] = canonical_sha256(
        {
            "git_commit": manifest["git"]["commit"],
            "model_sha256": model["sha256"],
            "fixed_test_sha256": fixed_test["sha256"],
            "evaluation_registry_sha256": evaluation_registry["sha256"],
            "config_sha256": manifest["config"]["resolved_sha256"],
            "prompt_sha256": prompts["combined_sha256"],
            "difficulty_sha256": manifest["difficulty"]["sha256"],
            "difficulty_artifact_sha256": (
                difficulty_artifact["sha256"]
                if difficulty_artifact
                else None
            ),
            "environment": stable_environment,
        }
    )
    return manifest
