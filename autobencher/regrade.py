"""Offline, immutable regrading of saved fixed-test predictions."""

from __future__ import annotations

import json
import re
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from autobencher.structured import answers_equivalent


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_versioned_json(path: Path, payload: Any) -> None:
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != serialized:
            raise FileExistsError(
                f"Refusing to overwrite different regrade artifact: {path}"
            )
        return
    path.write_text(serialized, encoding="utf-8")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _candidate_answer(row: Mapping[str, Any]) -> Any:
    parsed = row.get("parsed_response")
    if isinstance(parsed, Mapping) and parsed.get("final_answer") is not None:
        return parsed["final_answer"]
    for key in ("test_taker_response", "predicted_answer", "answer"):
        if row.get(key) is not None:
            return row[key]
    return None


def _gold_answer(row: Mapping[str, Any]) -> Any:
    for key in ("canonical_answer", "gold_answer", "reference_answer"):
        if row.get(key) is not None:
            return row[key]
    return None


def regrade_records(
    rows: list[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    evaluator_version: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output: list[dict[str, Any]] = []
    changes = {
        "false_positives": [],
        "false_negatives": [],
        "backend_disagreements": [],
        "wrong_to_correct": [],
        "correct_to_wrong": [],
    }
    unchanged_correct = 0
    unchanged_wrong = 0
    for index, source in enumerate(rows):
        row = dict(source)
        question_id = str(row.get("question_id") or f"row-{index + 1}")
        gold = _gold_answer(row)
        candidate = _candidate_answer(row)
        answer_type = str(row.get("answer_type") or "text")
        if gold is None or candidate is None:
            decision = {
                "equivalent": False,
                "status": "regrade_input_incomplete",
                "needs_review": True,
                "disagreement": False,
                "backend_results": {},
            }
        else:
            decision = answers_equivalent(
                gold,
                candidate,
                answer_type,
                config,
                tolerance=row.get("tolerance"),
            )
        original = bool(row.get("is_correct", False))
        regraded = bool(decision["equivalent"])
        if original and regraded:
            unchanged_correct += 1
        elif not original and not regraded:
            unchanged_wrong += 1
        elif original and not regraded:
            changes["false_positives"].append(question_id)
            changes["correct_to_wrong"].append(question_id)
        else:
            changes["false_negatives"].append(question_id)
            changes["wrong_to_correct"].append(question_id)
        if decision.get("disagreement"):
            changes["backend_disagreements"].append(question_id)
        row.update(
            {
                "original_is_correct": original,
                "is_correct": regraded,
                "regrade_evaluator_version": evaluator_version,
                "regrade_status": decision["status"],
                "regrade_needs_review": bool(decision.get("needs_review")),
                "regrade_backend_results": decision.get("backend_results", {}),
            }
        )
        output.append(row)
    summary = {
        "evaluator_version": evaluator_version,
        "question_count": len(rows),
        "original_correct": sum(bool(row.get("is_correct")) for row in rows),
        "regraded_correct": sum(bool(row["is_correct"]) for row in output),
        **changes,
        "unchanged_correct": unchanged_correct,
        "unchanged_wrong": unchanged_wrong,
    }
    return output, summary


def regrade_run(
    run_dir: str | Path,
    *,
    evaluator_version: str = "typed_equivalence_v2",
) -> dict[str, Any]:
    root = Path(run_dir).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Run directory does not exist: {root}")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", evaluator_version):
        raise ValueError("evaluator_version must be a safe filename token")
    config_path = root / "resolved_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing saved resolved config: {config_path}")
    config = _read_json(config_path)
    inputs = sorted(
        path
        for path in (root / "fixed_test").glob(
            "*/fixed_math.test_taker_inference.json"
        )
        if ".regraded_" not in path.name
    )
    # Compatibility for early/fixture runs that stored item rows in the
    # comparison artifact. Current production runs store a summary there and
    # item-level records in test_taker_inference.json.
    if not inputs:
        inputs = sorted(
            path
            for path in (root / "fixed_test").glob(
                "*/fixed_math.compare_answers.json"
            )
            if ".regraded_" not in path.name
        )
    if not inputs:
        raise FileNotFoundError(f"No fixed-test comparison artifacts under {root}")

    run_summary: dict[str, Any] = {
        "schema_version": "1.0",
        "evaluator_version": evaluator_version,
        "run_dir": str(root),
        "stages": {},
    }
    for source_path in inputs:
        original_sha = _file_sha256(source_path)
        rows = _read_json(source_path)
        if not isinstance(rows, list):
            raise ValueError(f"Expected a JSON list: {source_path}")
        regraded, summary = regrade_records(
            rows, config, evaluator_version=evaluator_version
        )
        source_stem = source_path.name.removesuffix(".json")
        output_path = source_path.with_name(
            f"{source_stem}.regraded_{evaluator_version}.json"
        )
        summary_path = source_path.with_name(
            f"fixed_math.regrade_summary_{evaluator_version}.json"
        )
        summary.update(
            {
                "source_path": str(source_path),
                "source_sha256": original_sha,
                "output_path": str(output_path),
            }
        )
        _write_versioned_json(output_path, regraded)
        _write_versioned_json(summary_path, summary)
        if _file_sha256(source_path) != original_sha:
            raise RuntimeError(f"Source artifact changed during regrade: {source_path}")
        run_summary["stages"][source_path.parent.name] = summary
    aggregate_path = root / f"regrade_summary_{evaluator_version}.json"
    _write_versioned_json(aggregate_path, run_summary)
    run_summary["summary_path"] = str(aggregate_path)
    return run_summary
