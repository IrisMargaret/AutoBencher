"""Prepare and freeze model-panel difficulty calibration artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from autobencher.difficulty_calibration import (
    _load_jsonl,
    calibrate_from_files,
    freeze_calibration,
    prepare_panel_schedule,
)
from autobencher.experiment import atomic_json, atomic_yaml
from autobencher.fingerprints import file_sha256


DATA_ROOT = Path(
    "/vepfs-mlp2/queue010/20262202597/math_flywheel"
).resolve()


def _safe_output(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(DATA_ROOT):
        raise SystemExit(f"Output must be beneath {DATA_ROOT}: {path}")
    return path


def _safe_data_input(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_relative_to(DATA_ROOT):
        raise SystemExit(f"Calibration data must be beneath {DATA_ROOT}: {path}")
    if not path.is_file():
        raise SystemExit(f"Calibration data file does not exist: {path}")
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare-panel")
    prepare.add_argument("--questions", required=True)
    prepare.add_argument("--panel-config", required=True)
    prepare.add_argument("--output", required=True)
    prepare.add_argument("--dataset-role", default="difficulty_calibration")
    calibrate = subparsers.add_parser("calibrate")
    calibrate.add_argument("--questions", required=True)
    calibrate.add_argument("--responses", required=True)
    calibrate.add_argument("--output", required=True)
    calibrate.add_argument("--dataset-role", default="difficulty_calibration")
    calibrate.add_argument("--freeze", action="store_true")
    calibrate.add_argument("--minimum-model-coverage", type=float, default=0.80)
    calibrate.add_argument("--minimum-models-per-item", type=int, default=3)
    calibrate.add_argument("--minimum-models-per-tier", type=int, default=1)
    calibrate.add_argument("--maximum-missing-rate", type=float, default=0.20)
    args = parser.parse_args()
    output = _safe_output(args.output)
    questions = _safe_data_input(args.questions)
    if args.command == "prepare-panel":
        panel = yaml.safe_load(
            Path(args.panel_config).read_text(encoding="utf-8")
        )
        tasks = prepare_panel_schedule(
            _load_jsonl(questions),
            panel["panel_models"],
            dataset_role=args.dataset_role,
        )
        atomic_json(
            {
                "schema_version": "1.0",
                "dataset_role": args.dataset_role,
                "tasks": tasks,
            },
            output,
        )
        print(json.dumps({"task_count": len(tasks), "output": str(output)}))
        return 0
    candidate = calibrate_from_files(
        questions,
        _safe_data_input(args.responses),
        dataset_role=args.dataset_role,
        minimum_model_coverage=args.minimum_model_coverage,
        minimum_models_per_item=args.minimum_models_per_item,
        minimum_models_per_tier=args.minimum_models_per_tier,
        maximum_missing_rate=args.maximum_missing_rate,
    )
    result = freeze_calibration(candidate, output) if args.freeze else candidate
    if not args.freeze:
        atomic_json(result, output)
    snippet_path = None
    if args.freeze:
        snippet_path = output.with_name(output.stem + ".config-snippet.yaml")
        atomic_yaml(
            {
                "difficulty": {
                    "rubric_version": "calibrated_math_v2",
                    "calibration_artifact": output.as_posix(),
                    "calibration_artifact_sha256": file_sha256(output),
                    "dimension_weights": result["weights"],
                }
            },
            snippet_path,
        )
    print(
        json.dumps(
            {
                "status": result["status"],
                "item_count": result["item_count"],
                "model_count": result["model_count"],
                "output": str(output),
                "artifact_sha256": file_sha256(output),
                "config_snippet": (
                    str(snippet_path) if snippet_path else None
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
