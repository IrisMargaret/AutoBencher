"""Create a reproducibility manifest for the current experiment baseline."""

# ruff: noqa: E402 -- bytecode suppression must precede project imports.

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.dont_write_bytecode = True

from autobencher.baseline import build_baseline_manifest
from autobencher.config import load_resolved_config
from autobencher.experiment import atomic_json


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path(
    "/vepfs-mlp2/queue010/20262202597/math_flywheel"
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Freeze code/config/model/test/prompt fingerprints."
    )
    parser.add_argument(
        "--config",
        default="configs/studies/full.yaml",
    )
    parser.add_argument(
        "--environment",
        default="configs/environments/volcengine.yaml",
    )
    parser.add_argument(
        "--baseline-id",
        default="baseline-ablation-v1",
    )
    parser.add_argument(
        "--output",
        default=str(
            DEFAULT_DATA_ROOT
            / "baselines"
            / "baseline-ablation-v1.json"
        ),
    )
    parser.add_argument(
        "--allow-missing-artifacts",
        action="store_true",
        help="Record unresolved paths/identifiers without claiming content verification.",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Allow a manifest to be written for a dirty worktree.",
    )
    args = parser.parse_args()

    config, provenance = load_resolved_config(
        PROJECT_ROOT / args.config,
        PROJECT_ROOT / args.environment if args.environment else None,
        validate_paths=False,
    )
    manifest = build_baseline_manifest(
        config,
        provenance,
        PROJECT_ROOT,
        baseline_id=args.baseline_id,
        allow_missing_artifacts=args.allow_missing_artifacts,
    )
    if manifest["git"]["dirty"] and not args.allow_dirty:
        changes = "\n".join(manifest["git"]["changes"][:20])
        raise SystemExit(
            "Refusing to freeze a dirty worktree. Commit the baseline first or "
            "pass --allow-dirty for a provisional manifest.\n"
            f"{changes}"
        )
    output = Path(args.output).expanduser()
    atomic_json(manifest, output)
    print(f"Baseline manifest: {output}")
    print(f"Baseline SHA256: {manifest['baseline_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
