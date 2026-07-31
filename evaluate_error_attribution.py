"""Export or score a human audit of evidence-backed error attribution."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from autobencher.attribution_eval import (
    evaluate_review_csv,
    export_blinded_review_packets,
    export_review_sample,
    merge_blinded_reviews,
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

    blind = subparsers.add_parser(
        "export-blinded",
        help="create two independent blinded packets and sealed predictions",
    )
    blind.add_argument("--input", required=True)
    blind.add_argument("--output-dir", required=True)
    blind.add_argument(
        "--sealed-output-dir",
        help="operator-only directory outside the annotator packet directory",
    )
    blind.add_argument("--sample-size", type=int, default=400)
    blind.add_argument("--seed", type=int, default=42)

    merge = subparsers.add_parser(
        "merge",
        help="merge completed blind packets for consensus/adjudication",
    )
    merge.add_argument("--system-predictions", required=True)
    merge.add_argument("--annotator-1", required=True)
    merge.add_argument("--annotator-2", required=True)
    merge.add_argument("--output", required=True)

    score = subparsers.add_parser(
        "score",
        help="score a completed review CSV",
    )
    score.add_argument("--review-csv", required=True)
    score.add_argument("--output", required=True)
    score.add_argument("--minimum-reviewed-count", type=int, default=300)
    score.add_argument("--minimum-kappa", type=float, default=0.70)
    score.add_argument("--minimum-high-confidence-accuracy", type=float, default=0.80)
    score.add_argument("--minimum-high-confidence-count", type=int, default=50)
    score.add_argument("--minimum-completion-rate", type=float, default=0.90)
    score.add_argument("--maximum-unresolved-rate", type=float, default=0.0)
    score.add_argument("--minimum-per-label-count", type=int, default=5)
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
    elif args.command == "export-blinded":
        input_path = _contained_path(args.input, allowed)
        output_dir = _contained_path(args.output_dir, allowed)
        result = export_blinded_review_packets(
            _read_records(input_path),
            output_dir,
            sample_size=args.sample_size,
            seed=args.seed,
            sealed_output_dir=(
                _contained_path(args.sealed_output_dir, allowed)
                if args.sealed_output_dir
                else None
            ),
        )
    elif args.command == "merge":
        result = merge_blinded_reviews(
            _contained_path(args.system_predictions, allowed),
            _contained_path(args.annotator_1, allowed),
            _contained_path(args.annotator_2, allowed),
            _contained_path(args.output, allowed),
        )
    else:
        review_path = _contained_path(args.review_csv, allowed)
        output_path = _contained_path(args.output, allowed)
        result = evaluate_review_csv(
            review_path,
            output_path,
            minimum_reviewed_count=args.minimum_reviewed_count,
            minimum_cohen_kappa=args.minimum_kappa,
            minimum_high_confidence_accuracy=(
                args.minimum_high_confidence_accuracy
            ),
            minimum_high_confidence_count=args.minimum_high_confidence_count,
            minimum_completion_rate=args.minimum_completion_rate,
            maximum_unresolved_rate=args.maximum_unresolved_rate,
            minimum_per_label_count=args.minimum_per_label_count,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
