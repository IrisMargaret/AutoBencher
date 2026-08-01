#!/usr/bin/env python3
"""Distill a 270-record generation guide from open-source test candidates."""

from __future__ import annotations

import argparse
import json

from autobencher.generation_guidance import write_generation_guidance


DEFAULT_DATA_ROOT = "/vepfs-mlp2/queue010/20262202597/math_flywheel"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-candidates",
        default=f"{DEFAULT_DATA_ROOT}/benchmarks/official_candidates_v1.json",
    )
    parser.add_argument(
        "--output",
        default=(
            f"{DEFAULT_DATA_ROOT}/guidance/"
            "open_source_generation_guidance_v1.json"
        ),
    )
    parser.add_argument("--allowed-data-root", default=DEFAULT_DATA_ROOT)
    parser.add_argument("--records-per-subcategory", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--similarity-threshold", type=float, default=0.72)
    args = parser.parse_args()
    manifest = write_generation_guidance(
        args.source_candidates,
        args.output,
        allowed_data_root=args.allowed_data_root,
        records_per_subcategory=args.records_per_subcategory,
        seed=args.seed,
        similarity_threshold=args.similarity_threshold,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
