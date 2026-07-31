#!/usr/bin/env python3
"""Regrade a completed run from saved answers, without model/API calls."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from autobencher.regrade import regrade_run


DEFAULT_DATA_ROOT = Path(
    "/vepfs-mlp2/queue010/20262202597/math_flywheel"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument(
        "--evaluator-version", default="typed_equivalence_v2"
    )
    parser.add_argument(
        "--allowed-data-root",
        default=os.environ.get("AUTOBENCHER_DATA_ROOT", str(DEFAULT_DATA_ROOT)),
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir).resolve()
    allowed_root = Path(args.allowed_data_root).resolve()
    if not run_dir.is_relative_to(allowed_root):
        parser.error(
            f"--run-dir must stay under --allowed-data-root ({allowed_root})"
        )
    summary = regrade_run(
        run_dir, evaluator_version=args.evaluator_version
    )
    compact = {
        "evaluator_version": summary["evaluator_version"],
        "summary_path": summary["summary_path"],
        "stages": {
            stage: {
                "question_count": item["question_count"],
                "original_correct": item["original_correct"],
                "regraded_correct": item["regraded_correct"],
                "wrong_to_correct": len(item["wrong_to_correct"]),
                "correct_to_wrong": len(item["correct_to_wrong"]),
            }
            for stage, item in summary["stages"].items()
        },
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
