#!/usr/bin/env python3
"""Prepare a blind set without printing its seed, questions, or answers."""

from __future__ import annotations

import argparse
import json
import os

from autobencher.blind_benchmark import prepare_blind_benchmark


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--contamination-data", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--allowed-data-root", required=True)
    parser.add_argument("--questions-per-subcategory", type=int, default=20)
    args = parser.parse_args()
    seed_file = os.environ.get("AUTOBENCHER_BLIND_SEED_FILE", "").strip()
    if not seed_file:
        parser.error("AUTOBENCHER_BLIND_SEED_FILE must be provisioned out-of-band")
    result = prepare_blind_benchmark(
        candidate_path=args.candidates,
        contamination_paths=args.contamination_data,
        seed_file=seed_file,
        output_path=args.output,
        manifest_path=args.manifest_output,
        allowed_data_root=args.allowed_data_root,
        questions_per_subcategory=args.questions_per_subcategory,
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "benchmark_role",
                    "question_count",
                    "subcategory_count",
                    "sha256",
                    "dedup_policy_version",
                    "dedup_rejections",
                )
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
