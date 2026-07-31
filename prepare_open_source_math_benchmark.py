#!/usr/bin/env python3
"""Build a balanced 540-item candidate pool from local upstream test files."""

from __future__ import annotations

import argparse
import json

from autobencher.open_source_benchmark import write_open_source_candidate_set


DEFAULT_DATA_ROOT = "/vepfs-mlp2/queue010/20262202597/math_flywheel"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--catalog",
        default="configs/open_source_evaluation_sources.yaml",
    )
    parser.add_argument(
        "--source-root",
        default=f"{DEFAULT_DATA_ROOT}/source_datasets",
    )
    parser.add_argument(
        "--output",
        default=f"{DEFAULT_DATA_ROOT}/benchmarks/official_candidates_v1.json",
    )
    parser.add_argument("--allowed-data-root", default=DEFAULT_DATA_ROOT)
    args = parser.parse_args()
    manifest = write_open_source_candidate_set(
        args.catalog,
        args.source_root,
        args.output,
        args.allowed_data_root,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
