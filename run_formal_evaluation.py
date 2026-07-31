"""Run official/blind evaluation only from a registry-bound checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

from autobencher.experiment import atomic_json, git_commit
from autobencher.fingerprints import artifact_fingerprint, file_sha256
from autobencher.result_schema import ExperimentRecord, validate_registry
from autobencher.study_runner import validate_experiment_completion


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _override(name, value):
    return f"{name}={json.dumps(value, ensure_ascii=False)}"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Checkpoint-bound official or blind evaluation runner."
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--environment")
    parser.add_argument("--source-index", required=True)
    parser.add_argument("--source-study-id", required=True)
    parser.add_argument("--source-method", required=True)
    parser.add_argument("--source-seed", required=True, type=int)
    parser.add_argument("--checkpoint-path", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    registry = _json(Path(args.source_index).expanduser().resolve())
    validate_registry(registry)
    matches = [
        ExperimentRecord.from_dict(item)
        for item in registry["experiments"]
        if item.get("study_id") == args.source_study_id
    ]
    if len(matches) != 1:
        raise SystemExit("source_study_id must identify exactly one registry run")
    source = matches[0]
    if source.status != "completed":
        raise SystemExit("source training run is not completed")
    if source.method != args.source_method or source.seed != args.source_seed:
        raise SystemExit("source method/seed does not match the registry")
    completion = validate_experiment_completion(source)
    checkpoint = Path(args.checkpoint_path).expanduser().resolve()
    observed_sha = artifact_fingerprint(checkpoint, allow_missing=False)["sha256"]
    if observed_sha != args.checkpoint_sha256.lower():
        raise SystemExit("explicit checkpoint hash does not match checkpoint_path")
    if observed_sha != completion["checkpoint_sha256"]:
        raise SystemExit("checkpoint hash does not match source registry binding")
    if source.method != "base":
        manifests = sorted(
            Path(source.run_dir).glob(
                "cycle/cycle_*/training/checkpoint_manifest.json"
            )
        )
        bound_path = Path(
            str(_json(manifests[-1])["merged_model_path"])
        ).resolve()
        if checkpoint != bound_path:
            raise SystemExit("checkpoint_path is not the source run's selected model")

    raw_config = yaml.safe_load(
        Path(args.config).expanduser().resolve().read_text(encoding="utf-8")
    )
    active_set = str(
        (raw_config.get("evaluation_sets") or {}).get("active_set", "")
    )
    blind_receipt_path = None
    blind_receipt = None
    if active_set == "blind_final_v1":
        benchmark_sha = os.environ.get(
            "AUTOBENCHER_BLIND_TEST_SHA256", ""
        ).strip().lower()
        if len(benchmark_sha) != 64:
            raise SystemExit("blind benchmark SHA-256 is not provisioned")
        data_root = Path(
            os.environ.get(
                "AUTOBENCHER_DATA_ROOT",
                "/vepfs-mlp2/queue010/20262202597/math_flywheel",
            )
        ).expanduser().resolve()
        receipt_dir = data_root / "blind_receipts"
        receipt_dir.mkdir(parents=True, exist_ok=True)
        blind_receipt_path = receipt_dir / f"{observed_sha}-{benchmark_sha}.json"
        if blind_receipt_path.exists():
            raise SystemExit(
                "this model snapshot and blind benchmark have already been evaluated"
            )
        blind_receipt = {
            "schema_version": "1.0",
            "status": "running",
            "model_sha256": observed_sha,
            "benchmark_sha256": benchmark_sha,
            "config_sha256": file_sha256(Path(args.config).resolve()),
            "git_commit": git_commit(Path(__file__).resolve().parent),
            "run_id": args.run_id,
            "started_at": _utc_now(),
        }
        atomic_json(blind_receipt, blind_receipt_path)
        if os.name != "nt":
            os.chmod(blind_receipt_path, 0o600)

    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve().parent / "run_scripts.py"),
        "math",
        "--config",
        str(Path(args.config).expanduser().resolve()),
    ]
    if args.environment:
        command.extend(
            ["--environment", str(Path(args.environment).expanduser().resolve())]
        )
    command.extend(
        [
            "--run-id",
            args.run_id,
            "--test-taker-modelname",
            str(checkpoint),
            "--override",
            _override("evaluation_provenance.require_checkpoint_binding", True),
            _override("evaluation_provenance.execution_policy", "eval_only"),
            _override("evaluation_provenance.evaluated_method", source.method),
            _override("evaluation_provenance.source_study_id", source.study_id),
            _override("evaluation_provenance.source_run_id", source.study_id),
            _override("evaluation_provenance.source_seed", source.seed),
            _override("evaluation_provenance.checkpoint_path", str(checkpoint)),
            _override("evaluation_provenance.checkpoint_sha256", observed_sha),
            _override("models.test_taker.model_path", str(checkpoint)),
        ]
    )
    print(subprocess.list2cmdline(command))
    if args.dry_run:
        if blind_receipt_path is not None:
            blind_receipt_path.unlink(missing_ok=True)
        return 0
    return_code = subprocess.run(
        command,
        cwd=Path(__file__).resolve().parent,
        check=False,
    ).returncode
    if blind_receipt_path is not None and blind_receipt is not None:
        atomic_json(
            {
                **blind_receipt,
                "status": "completed" if return_code == 0 else "failed",
                "completed_at": _utc_now(),
                "return_code": return_code,
            },
            blind_receipt_path,
        )
        if os.name != "nt":
            os.chmod(blind_receipt_path, 0o600)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
