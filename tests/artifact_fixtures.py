"""Strict, registry-bound experiment artifacts used by integration tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from autobencher.budget_ledger import BudgetLedger
from autobencher.fingerprints import (
    artifact_fingerprint,
    canonical_sha256,
    file_sha256,
)
from autobencher.result_schema import ExperimentRecord


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_completed_run(
    record: ExperimentRecord,
    baseline_answers: Sequence[bool],
    final_answers: Sequence[bool] | None = None,
    *,
    resolved_config: Mapping[str, Any] | None = None,
) -> None:
    """Create the smallest artifact graph accepted by the production validator."""
    run_dir = Path(record.run_dir or Path(record.experiment_dir) / "bound_run")
    run_dir.mkdir(parents=True, exist_ok=True)
    record.run_dir = str(run_dir.resolve())
    record.run_manifest_path = str((run_dir / "run_manifest.json").resolve())
    record.summary_path = str((run_dir / "experiment_summary.json").resolve())
    record.artifact_manifest_path = str(
        (run_dir / "artifact_manifest.json").resolve()
    )

    resolved = dict(resolved_config or {"fixture": record.study_id})
    resolved_sha = canonical_sha256(resolved)
    if resolved_config is not None and record.config_hash != resolved_sha:
        raise AssertionError("Fixture resolved config does not match registry hash")
    record.config_hash = resolved_sha
    existing_base_sha = record.fingerprints.get("base_model", {}).get("sha256")
    if existing_base_sha:
        base_sha = existing_base_sha
    else:
        base_model = run_dir / "fixture_base_model"
        base_model.mkdir(exist_ok=True)
        (base_model / "config.json").write_text(
            '{"model_type":"fixture"}\n', encoding="utf-8"
        )
        base_sha = artifact_fingerprint(base_model)["sha256"]
    fixed_sha = record.fingerprints.get("fixed_test", {}).get("sha256") or (
        canonical_sha256(
            {"evaluation": "fixture-v1", "count": len(baseline_answers)}
        )
    )
    prompt_sha = record.fingerprints.get("prompt_bundle", {}).get(
        "combined_sha256", "fixture-prompt-sha"
    )
    guidance_sha = record.fingerprints.get("generation_guidance", {}).get(
        "sha256", canonical_sha256({"enabled": False})
    )
    record.fingerprints = {
        **record.fingerprints,
        "base_model": {"sha256": base_sha},
        "fixed_test": {"sha256": fixed_sha},
        "prompt_bundle": {"combined_sha256": prompt_sha},
        "generation_guidance": {"sha256": guidance_sha},
    }

    questions = [
        {"question_id": f"q{index}"}
        for index in range(1, len(baseline_answers) + 1)
    ]
    _write_json(
        run_dir / "fixed_test" / "dataset_snapshot.json",
        {
            "evaluation_set_id": "fixture_fixed",
            "evaluation_version": "1.0",
            "sha256": fixed_sha,
            "question_count": len(questions),
            "questions": questions,
        },
    )

    stages = [("baseline", baseline_answers)]
    if record.method != "base":
        stages.append(("cycle_1", final_answers or baseline_answers))
    for stage, answers in stages:
        rows = [
            {
                "question_id": f"q{index}",
                "category": "Algebra",
                "sub_category": "Linear Equations",
                "difficulty": index,
                "gold_answer": "1",
                "test_taker_response": "1" if correct else "0",
                "is_correct": bool(correct),
                "primary_error_tag": (
                    None if correct else "calculation_error"
                ),
                "parsed_response": {"confidence": 0.8},
                "latency": 0.1,
                "input_tokens": 10,
                "output_tokens": 2,
            }
            for index, correct in enumerate(answers, start=1)
        ]
        _write_json(
            run_dir
            / "fixed_test"
            / stage
            / "fixed_math.compare_answers.json",
            rows,
        )

    final = list(final_answers or baseline_answers)
    baseline_accuracy = sum(baseline_answers) / len(baseline_answers)
    final_accuracy = sum(final) / len(final)
    identity = {
        "run_id": record.study_id,
        "config_hash": record.config_hash,
        "git_commit": record.git_commit,
    }
    _write_json(run_dir / "resolved_config.json", resolved)
    _write_json(
        run_dir / "cycle_record.json",
        {**identity, "status": "completed", "cycles": []},
    )
    _write_json(
        run_dir / "run_manifest.json",
        {
            **identity,
            "status": "completed",
            "prompt_hash": prompt_sha,
            "base_model_sha256": base_sha,
            "generation_guidance_sha256": guidance_sha,
        },
    )
    _write_json(
        run_dir / "experiment_summary.json",
        {
            **identity,
            "status": "completed",
            "baseline_accuracy": baseline_accuracy,
            "final_accuracy": final_accuracy,
            "accuracy_delta": final_accuracy - baseline_accuracy,
            "total_questions": len(baseline_answers),
        },
    )
    ledger = BudgetLedger(
        run_dir / "budget_ledger.json",
        {"protocol": record.budget_protocol, "pricing": {}},
        identity,
    )
    ledger.record_accuracy(baseline_accuracy, final_accuracy)
    ledger.finalize("completed")

    if record.method != "base":
        model_dir = run_dir / "cycle" / "cycle_1" / "model"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / "config.json").write_text(
            '{"model_type":"trained-fixture"}\n', encoding="utf-8"
        )
        (model_dir / "model.safetensors").write_bytes(b"fixture-weights")
        checkpoint_path = (
            run_dir
            / "cycle"
            / "cycle_1"
            / "training"
            / "checkpoint_manifest.json"
        )
        _write_json(
            checkpoint_path,
            {
                **identity,
                "merged_model_path": str(model_dir.resolve()),
                "merged_model_sha256": artifact_fingerprint(model_dir)[
                    "sha256"
                ],
            },
        )
        record.checkpoint_sha256 = artifact_fingerprint(model_dir)["sha256"]
    else:
        record.checkpoint_sha256 = base_sha

    artifact_paths = [
        run_dir / "run_manifest.json",
        run_dir / "experiment_summary.json",
        run_dir / "budget_ledger.json",
        run_dir / "cycle_record.json",
        run_dir / "resolved_config.json",
        run_dir / "fixed_test" / "dataset_snapshot.json",
        *run_dir.glob("fixed_test/*/fixed_math.compare_answers.json"),
        *run_dir.glob("cycle/cycle_*/training/checkpoint_manifest.json"),
    ]
    _write_json(
        run_dir / "artifact_manifest.json",
        {
            **identity,
            "files": [
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "sha256": file_sha256(path),
                    "size": path.stat().st_size,
                }
                for path in artifact_paths
            ],
        },
    )
    record.artifact_manifest_sha256 = file_sha256(
        run_dir / "artifact_manifest.json"
    )
    record.evaluation_set_id = "fixture_fixed"
    record.evaluation_set_version = "1.0"
    record.evaluation_set_sha256 = fixed_sha
