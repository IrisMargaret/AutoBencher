"""Offline, read-only repository and configuration health check."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Callable

import yaml

from autobencher.config import load_project_config
from autobencher.storage import resolved_storage_paths


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path("/vepfs-mlp2/queue010/20262202597/math_flywheel")
FORMAL_CONFIGS = {"official_fixed_eval.yaml", "blind_final.yaml"}
FORMAL_OVERRIDES = {
    "evaluation_provenance": {
        "evaluated_method": "full",
        "source_study_id": "health-check-study",
        "source_run_id": "health-check-run",
        "source_seed": 42,
        "checkpoint_path": f"{DATA_ROOT.as_posix()}/models/health-check",
        "checkpoint_sha256": "0" * 64,
    }
}


def _python_sources() -> list[Path]:
    roots = [PROJECT_ROOT / "autobencher", PROJECT_ROOT / "tests", PROJECT_ROOT / "tools"]
    sources = [path for root in roots for path in root.rglob("*.py")]
    sources.extend(PROJECT_ROOT.glob("*.py"))
    return sorted(set(sources))


def _check_python_syntax() -> dict[str, Any]:
    sources = _python_sources()
    for path in sources:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")
    return {"source_count": len(sources), "bytecode_written": False}


def _check_structured_files() -> dict[str, Any]:
    counts = {"json": 0, "yaml": 0}
    for path in sorted(PROJECT_ROOT.rglob("*")):
        if not path.is_file() or any(part.startswith(".") for part in path.parts):
            continue
        if path.suffix == ".json":
            json.loads(path.read_text(encoding="utf-8"))
            counts["json"] += 1
        elif path.suffix in {".yaml", ".yml"}:
            yaml.safe_load(path.read_text(encoding="utf-8"))
            counts["yaml"] += 1
    return counts


def _check_runtime_configs() -> dict[str, Any]:
    paths = sorted((PROJECT_ROOT / "configs" / "experiments").glob("*.yaml"))
    paths.extend(
        sorted((PROJECT_ROOT / "configs" / "studies").glob("*.yaml"))
    )
    paths.extend(
        PROJECT_ROOT / "configs" / name
        for name in (
            "math_flywheel.yaml",
            "math_flywheel_local.yaml",
            "math_flywheel_smoke_test.yaml",
            "math_flywheel_volcengine.yaml",
        )
    )
    hashes = {}
    for path in paths:
        overrides = FORMAL_OVERRIDES if path.name in FORMAL_CONFIGS else None
        _, provenance = load_project_config(
            path,
            cli_overrides=overrides,
            validate_paths=False,
        )
        hashes[path.relative_to(PROJECT_ROOT).as_posix()] = provenance["config_hash"]
    return {"config_count": len(hashes), "config_hashes": hashes}


def _check_storage_contract() -> dict[str, Any]:
    config, _ = load_project_config(
        PROJECT_ROOT / "configs" / "experiments" / "math_flywheel.yaml",
        environment_path=PROJECT_ROOT / "configs" / "environments" / "volcengine.yaml",
        validate_paths=False,
    )
    paths = resolved_storage_paths(config)
    allowed = Path(str(config["paths"]["allowed_data_root"])).resolve()
    escaped = {
        name: path.as_posix()
        for name, path in paths.items()
        if not path.is_relative_to(allowed)
    }
    if escaped:
        raise RuntimeError(f"Runtime storage escapes allowed root: {escaped}")
    return {
        "allowed_data_root": allowed.as_posix(),
        "paths": {name: path.as_posix() for name, path in paths.items()},
    }


def _check_study_suites() -> dict[str, Any]:
    summaries = {}
    for path in sorted((PROJECT_ROOT / "configs" / "study_suites").glob("*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        suite = payload["study_suite"]
        environment = PROJECT_ROOT / str(suite["environment"])
        if not environment.is_file():
            raise FileNotFoundError(f"Study suite environment is missing: {environment}")
        methods = [str(value) for value in suite["methods"]]
        if len(methods) != len(set(methods)):
            raise ValueError(f"Study suite contains duplicate methods: {path}")
        if not suite.get("seeds") or not suite.get("models") or not suite.get("budgets"):
            raise ValueError(f"Study suite has an empty matrix axis: {path}")
        output_root = Path(str(suite["output_root"]))
        if not output_root.is_relative_to(DATA_ROOT):
            raise ValueError(f"Study output escapes VEPFS data root: {path}")
        summaries[path.name] = {
            "methods": len(methods),
            "seeds": len(suite["seeds"]),
            "models": len(suite["models"]),
            "budgets": len(suite["budgets"]),
        }
    return {"suite_count": len(summaries), "suites": summaries}


def _check_benchmarks() -> dict[str, Any]:
    summaries = {}
    forbidden_sources = {"gsm8k", "hendrycks_math", "mmlu", "huggingface"}
    for path in sorted((PROJECT_ROOT / "benchmarks").glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        questions = payload.get("questions", [])
        identifiers = [
            str(item.get("question_id", item.get("id", ""))).strip()
            for item in questions
        ]
        if not questions or not all(identifiers) or len(identifiers) != len(set(identifiers)):
            raise ValueError(f"Benchmark has missing/duplicate question IDs: {path}")
        observed_sources = {
            str(item.get("source_dataset", "project_native")).strip().lower()
            for item in questions
        }
        if observed_sources & forbidden_sources:
            raise ValueError(f"Forbidden benchmark provenance in {path}")
        summaries[path.name] = {
            "question_count": len(questions),
            "source_datasets": sorted(observed_sources),
        }
    return summaries


def _check_repository_hygiene() -> dict[str, Any]:
    secret_patterns = {
        "literal_client_secret": re.compile(
            r"client_secret\s*=\s*['\"][^'\"]{8,}['\"]",
            re.IGNORECASE,
        ),
        "literal_bearer_token": re.compile(
            r"access_token\s*=\s*['\"][A-Za-z0-9_.-]{40,}['\"]",
            re.IGNORECASE,
        ),
    }
    findings = []
    for path in _python_sources():
        text = path.read_text(encoding="utf-8")
        for name, pattern in secret_patterns.items():
            if pattern.search(text):
                findings.append(f"{name}:{path.relative_to(PROJECT_ROOT).as_posix()}")
    if findings:
        raise RuntimeError("Literal credential-like values found: " + ", ".join(findings))
    dependency_names = {}
    for filename in ("requirements.txt", "requirements-lock.txt"):
        names = []
        for line in (PROJECT_ROOT / filename).read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            name = re.split(r"[<>=!~\[]", stripped, maxsplit=1)[0].strip().lower()
            names.append(name)
        if len(names) != len(set(names)):
            raise RuntimeError(f"{filename} contains duplicate dependency declarations")
        dependency_names[filename] = set(names)
    missing_locks = sorted(
        dependency_names["requirements.txt"]
        - dependency_names["requirements-lock.txt"]
    )
    if missing_locks:
        raise RuntimeError(f"Direct dependencies missing from lock file: {missing_locks}")
    for readme_name in ("README.md", "README.zh-CN.md"):
        text = (PROJECT_ROOT / readme_name).read_text(encoding="utf-8")
        required = (
            "python smoke_test.py",
            "python -m pytest -q",
            "python run_study.py",
            "python run_formal_evaluation.py",
            "--generation-data",
        )
        missing = [value for value in required if value not in text]
        if missing:
            raise RuntimeError(f"{readme_name} lacks required operations: {missing}")
    return {
        "literal_credentials_found": 0,
        "readmes_checked": 2,
        "direct_dependency_count": len(dependency_names["requirements.txt"]),
        "locked_dependency_count": len(dependency_names["requirements-lock.txt"]),
    }


CHECKS: tuple[tuple[str, Callable[[], dict[str, Any]]], ...] = (
    ("python_syntax", _check_python_syntax),
    ("structured_files", _check_structured_files),
    ("runtime_configs", _check_runtime_configs),
    ("storage_contract", _check_storage_contract),
    ("study_suites", _check_study_suites),
    ("benchmarks", _check_benchmarks),
    ("repository_hygiene", _check_repository_hygiene),
)


def verify_project() -> dict[str, Any]:
    results = {}
    failures = []
    for name, check in CHECKS:
        try:
            results[name] = {"status": "passed", **check()}
        except Exception as exc:  # report all independent checks in one pass
            results[name] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
            failures.append(name)
    return {
        "status": "passed" if not failures else "failed",
        "project_root": PROJECT_ROOT.as_posix(),
        "failures": failures,
        "checks": results,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Run offline syntax, config, storage, benchmark, and hygiene checks."
    )
    parser.parse_args(argv)
    report = verify_project()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
