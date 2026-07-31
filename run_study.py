"""Command-line entry point for deterministic ablation study suites."""

# ruff: noqa: E402 -- bytecode suppression must precede project imports.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# The user-data policy forbids repository-local runtime artifacts.
sys.dont_write_bytecode = True

from autobencher.study_runner import StudyRunner


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run or resume a method × seed × model × budget study."
    )
    parser.add_argument(
        "--suite",
        required=True,
        help="Study-suite YAML, for example configs/study_suites/smoke.yaml.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume pending/failed/partial experiments and skip completed ones.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate fairness and write the experiment index without executing.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Development only: allow an uncommitted worktree.",
    )
    parser.add_argument("--method", action="append", default=[])
    parser.add_argument("--study-id", action="append", default=[])
    parser.add_argument("--max-experiments", type=int)
    args = parser.parse_args()
    if args.max_experiments is not None and args.max_experiments < 1:
        parser.error("--max-experiments must be positive")

    runner = StudyRunner(
        args.suite,
        project_root=PROJECT_ROOT,
        allow_dirty_worktree=args.allow_dirty,
    )
    records = runner.run(
        resume=args.resume,
        dry_run=args.dry_run,
        methods=set(args.method) or None,
        study_ids=set(args.study_id) or None,
        max_experiments=args.max_experiments,
    )
    counts = {}
    for record in records:
        counts[record.status] = counts.get(record.status, 0) + 1
    print(
        json.dumps(
            {
                "suite": runner.name,
                "experiment_index": str(runner.index_path),
                "experiment_count": len(records),
                "status_counts": counts,
                "dry_run": args.dry_run,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
