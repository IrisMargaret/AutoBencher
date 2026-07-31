"""Content-derived fingerprints for prompts, models, and benchmark artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


PROMPT_RELATIVE_PATHS = {
    "generator": "prompts/generator_question.txt",
    "test_taker": "prompts/test_taker.txt",
}


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256_bytes(payload.encode("utf-8"))


def resolve_project_path(project_root: str | Path, raw_path: str | Path) -> Path:
    path = Path(str(raw_path)).expanduser()
    if not path.is_absolute():
        path = Path(project_root) / path
    return path.resolve()


def _portable_path(path: Path, project_root: Path) -> str:
    try:
        return path.relative_to(project_root).as_posix()
    except ValueError:
        return path.as_posix()


def prompt_bundle_snapshot(
    config: Mapping[str, Any],
    project_root: str | Path,
    *,
    prompt_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Hash the actual prompt file bytes used by the three critical roles."""
    root = Path(project_root).resolve()
    declared = dict(PROMPT_RELATIVE_PATHS)
    declared["semantic_judge"] = str(
        config["evaluator_pipeline"]["semantic_judge_prompt_path"]
    )
    if prompt_paths:
        declared.update(prompt_paths)

    entries: dict[str, dict[str, str]] = {}
    combined_inputs = []
    for role in ("generator", "test_taker", "semantic_judge"):
        path = resolve_project_path(root, declared[role])
        if not path.is_file():
            raise FileNotFoundError(
                f"Prompt file for {role!r} does not exist: {path}"
            )
        portable = _portable_path(path, root)
        digest = file_sha256(path)
        entries[role] = {"path": portable, "sha256": digest}
        combined_inputs.append(
            {"role": role, "path": portable, "sha256": digest}
        )
    return {
        **entries,
        "combined_sha256": canonical_sha256(combined_inputs),
    }


def artifact_fingerprint(
    raw_path_or_identifier: str | Path,
    *,
    project_root: str | Path | None = None,
    allow_missing: bool = False,
) -> dict[str, Any]:
    """Fingerprint one file/tree, or explicitly mark an unresolved identifier."""
    raw = str(raw_path_or_identifier)
    candidate = Path(raw).expanduser()
    if project_root is not None and not candidate.is_absolute():
        project_candidate = Path(project_root) / candidate
        if project_candidate.exists():
            candidate = project_candidate

    if candidate.is_file():
        resolved = candidate.resolve()
        return {
            "kind": "file",
            "path": resolved.as_posix(),
            "sha256": file_sha256(resolved),
            "size_bytes": resolved.stat().st_size,
        }
    if candidate.is_dir():
        resolved = candidate.resolve()
        files = []
        total_size = 0
        for path in sorted(item for item in resolved.rglob("*") if item.is_file()):
            relative = path.relative_to(resolved).as_posix()
            size = path.stat().st_size
            total_size += size
            files.append(
                {
                    "path": relative,
                    "sha256": file_sha256(path),
                    "size_bytes": size,
                }
            )
        return {
            "kind": "directory",
            "path": resolved.as_posix(),
            "sha256": canonical_sha256(files),
            "file_count": len(files),
            "size_bytes": total_size,
        }

    looks_like_path = (
        candidate.is_absolute()
        or raw.startswith((".", "/", "\\"))
        or "\\" in raw
    )
    if looks_like_path and not allow_missing:
        raise FileNotFoundError(f"Artifact does not exist: {candidate}")
    return {
        "kind": "missing_path" if looks_like_path else "identifier",
        "path": candidate.as_posix() if looks_like_path else raw,
        "sha256": canonical_sha256(
            {
                "kind": "missing_path" if looks_like_path else "identifier",
                "value": candidate.as_posix() if looks_like_path else raw,
            }
        ),
        "content_verified": False,
    }
