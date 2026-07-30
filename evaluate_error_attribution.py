"""Export or score a human audit of evidence-backed error attribution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from autobencher.attribution_eval import (
    evaluate_review_csv,
    export_review_sample,
)
from autobencher.config import REQUIRED_DATA_ROOT


def _contained_path(value: str, allowed_root: str) -> Path:
    path = Path(value).expanduser().resolve()
    allowed = Path(allowed_root).expanduser().resolve()
    if not path.is_relative_to(allowed):
        raise RuntimeError(
            f"Audit data must remain below allowed data root {allowed}: {path}"
        )
    return path


def _read_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix.lower() == ".jsonl":
        return [
            json.loads(line)
            for line in text.splitlines()
            if line.strip()
        ]
    payload = json.loads(text)
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in (
            "records",
            "questions",
            "error_attributions",
            "generated_questions",
        ):
            if isinstance(payload.get(key), list):
                return [
                    item
                    for item in payload[key]
                    if isinstance(item, dict)
                ]
    raise ValueError(
        "Input must be a JSON array, JSONL records, or an object containing "
        "records/questions/error_attributions/generated_questions"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Human calibration for math error attribution",
    )
    parser.add_argument(
        "--allowed-data-root",
        default=os.environ.get(
            "AUTOBENCHER_ALLOWED_DATA_ROOT",
            REQUIRED_DATA_ROOT,
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    export = subparsers.add_parser(
        "export",
        help="sample incorrect records into a double-review CSV",
    )
    export.add_argument("--input", required=True)
    export.add_argument("--output", required=True)
    export.add_argument("--sample-size", type=int, default=100)
    export.add_argument("--seed", type=int, default=42)

    score = subparsers.add_parser(
        "score",
        help="score a completed review CSV",
    )
    score.add_argument("--review-csv", required=True)
    score.add_argument("--output", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    allowed = args.allowed_data_root
    if args.command == "export":
        input_path = _contained_path(args.input, allowed)
        output_path = _contained_path(args.output, allowed)
        count = export_review_sample(
            _read_records(input_path),
            output_path,
            sample_size=args.sample_size,
            seed=args.seed,
        )
        result = {
            "status": "completed",
            "review_sample_count": count,
            "output": output_path.as_posix(),
        }
    else:
        review_path = _contained_path(args.review_csv, allowed)
        output_path = _contained_path(args.output, allowed)
        result = evaluate_review_csv(review_path, output_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
