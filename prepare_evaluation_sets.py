"""Audit development questions and assemble immutable paper evaluation sets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from autobencher.config import load_project_config
from autobencher.evaluation_audit import (
    assemble_official_set,
    audit_evaluation_questions,
    load_json_questions,
)
from autobencher.experiment import atomic_json
from autobencher.fingerprints import file_sha256
from autobencher.fixed_benchmark import load_fixed_test_set
from autobencher.truth_solver import TruthSolver


DEFAULT_DATA_ROOT = (
    "/vepfs-mlp2/queue010/20262202597/math_flywheel"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit-development")
    audit.add_argument("--config", default="configs/math_flywheel.yaml")
    audit.add_argument(
        "--output",
        default=f"{DEFAULT_DATA_ROOT}/benchmarks/"
        "development_regression_v3.audit.json",
    )
    audit.add_argument("--training-data")

    assemble = subparsers.add_parser("assemble-official")
    assemble.add_argument("--candidates", required=True)
    assemble.add_argument(
        "--output",
        default=f"{DEFAULT_DATA_ROOT}/benchmarks/official_fixed_v1.json",
    )
    assemble.add_argument("--allowed-data-root", default=DEFAULT_DATA_ROOT)
    assemble.add_argument("--name", default="autobencher_official_fixed_v1")
    assemble.add_argument("--version", default="1.0")
    assemble.add_argument("--minimum-per-subcategory", type=int, default=20)
    assemble.add_argument("--max-per-template-cluster", type=int, default=2)
    assemble.add_argument("--near-duplicate-threshold", type=float, default=0.92)
    assemble.add_argument(
        "--training-data",
        required=True,
        help="Final training corpus used for leakage rejection.",
    )
    assemble.add_argument(
        "--generation-data",
        required=True,
        help="All raw/accepted generation records considered during development.",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "assemble-official":
        output_path = Path(args.output).expanduser().resolve()
        allowed_root = Path(args.allowed_data_root).expanduser().resolve()
        if not output_path.is_relative_to(allowed_root):
            raise ValueError(
                f"Official output and audit must be beneath {allowed_root}"
            )
        candidates = load_json_questions(args.candidates)
        training = load_json_questions(args.training_data)
        generation = load_json_questions(args.generation_data)
        config, _ = load_project_config(
            "configs/math_flywheel.yaml",
            validate_paths=False,
        )
        audit = audit_evaluation_questions(
            candidates,
            normalization_config=config,
            solver=TruthSolver.from_config(config),
            training_records=[*training, *generation],
            require_embedding_audit=True,
        )
        leaking = [
            item["question_id"]
            for item in audit["items"]
            if item["training_leakage_flags"]
        ]
        if leaking:
            raise ValueError(
                "Official candidates overlap training/generation corpora: "
                + ", ".join(leaking[:20])
            )
        if not audit["all_independently_verified"]:
            raise ValueError(
                "Official candidates contain unresolved independent-solver conflicts"
            )
        audit_path = output_path.with_suffix(".leakage_audit.json")
        atomic_json(audit, audit_path)
        result = assemble_official_set(
            candidates,
            name=args.name,
            version=args.version,
            output_path=args.output,
            allowed_data_root=args.allowed_data_root,
            minimum_questions_per_subcategory=args.minimum_per_subcategory,
            max_questions_per_template_cluster=args.max_per_template_cluster,
            near_duplicate_threshold=args.near_duplicate_threshold,
            training_leakage_audit={
                "training_record_count": len(training),
                "generation_record_count": len(generation),
                "leaking_question_count": 0,
                "training_corpus_sha256": file_sha256(args.training_data),
                "generation_corpus_sha256": file_sha256(args.generation_data),
                "candidate_corpus_sha256": file_sha256(args.candidates),
                "audit_algorithm_version": audit["audit_algorithm_version"],
                "thresholds": audit["thresholds"],
                "audit_report_sha256": file_sha256(audit_path),
                "audit_report_path": audit_path.as_posix(),
                "all_independently_verified": True,
            },
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0

    project_root = Path(__file__).resolve().parent
    config, _ = load_project_config(
        project_root / args.config,
        validate_paths=False,
    )
    questions, metadata = load_fixed_test_set(config, project_root)
    training = (
        load_json_questions(args.training_data)
        if args.training_data
        else []
    )
    audit = audit_evaluation_questions(
        questions,
        normalization_config=config,
        solver=TruthSolver.from_config(config),
        training_records=training,
    )
    output = Path(args.output).expanduser().resolve()
    allowed = Path(DEFAULT_DATA_ROOT).resolve()
    if not output.is_relative_to(allowed):
        raise ValueError(f"Audit output must be beneath {allowed}")
    atomic_json(
        {"dataset": metadata, **audit},
        output,
    )
    print(json.dumps({"output": output.as_posix(), **audit}, ensure_ascii=False, indent=2))
    return 0 if audit["all_independently_verified"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
