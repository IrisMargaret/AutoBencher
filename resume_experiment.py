"""Resume one existing Registry-bound experiment in its original directory."""

# ruff: noqa: E402 -- bytecode suppression must precede project imports.

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from autobencher.study_runner import resume_registered_experiment


PROJECT_ROOT = Path(__file__).resolve().parent


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Resume exactly one existing experiment from its Registry record; "
            "the suite matrix is not rebuilt."
        )
    )
    parser.add_argument("--registry", required=True)
    parser.add_argument("--study-id", required=True)
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow an audited compatibility patch on the registered commit.",
    )
    args = parser.parse_args()
    record = resume_registered_experiment(
        args.registry,
        args.study_id,
        project_root=PROJECT_ROOT,
        allow_dirty_worktree=args.allow_dirty,
    )
    print(
        json.dumps(
            {
                "study_id": record.study_id,
                "run_dir": record.run_dir,
                "status": record.status,
                "return_code": record.return_code,
                "error": record.error,
                "resume_source_root": record.resume_source_root,
                "runtime_patch_sha256": record.runtime_patch_sha256,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if record.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
