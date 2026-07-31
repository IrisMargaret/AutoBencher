"""Empirical calibration for the observable mathematics difficulty rubric.

The calibration set is deliberately separate from fixed and blind evaluation
sets.  Item difficulty is estimated from a heterogeneous model response panel,
then compared with the frozen structural rubric.  No fixed-test correctness
signal is accepted by this module.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .difficulty import DEFAULT_WEIGHTS
from .experiment import atomic_json, utc_now
from .fingerprints import canonical_sha256, file_sha256


CALIBRATED_RUBRIC_VERSION = "calibrated_math_v2"
FORBIDDEN_DATASET_ROLES = {
    "blind",
    "blind_test",
    "formal_fixed_test",
    "official_fixed_test",
    "final_blind_test",
}
DIMENSIONS = tuple(DEFAULT_WEIGHTS)


class DifficultyCalibrationError(ValueError):
    """Raised when calibration would be invalid or leak evaluation data."""


def _question_id(row: Mapping[str, Any]) -> str:
    value = row.get("question_id", row.get("id"))
    if value is None or not str(value).strip():
        raise DifficultyCalibrationError("Every row needs a question_id.")
    return str(value)


def _is_correct(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "yes"}:
        return True
    if normalized in {"false", "0", "no"}:
        return False
    raise DifficultyCalibrationError(f"Invalid is_correct value: {value!r}")


def _assert_calibration_role(role: str) -> None:
    normalized = role.strip().lower()
    if normalized in FORBIDDEN_DATASET_ROLES or "blind" in normalized:
        raise DifficultyCalibrationError(
            "Difficulty calibration cannot consume a blind or official fixed "
            f"evaluation set (dataset_role={role!r})."
        )
    if normalized not in {"difficulty_calibration", "calibration"}:
        raise DifficultyCalibrationError(
            "dataset_role must be difficulty_calibration; this explicit role "
            "prevents accidental fixed-test leakage."
        )


def _load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    result = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise DifficultyCalibrationError(
                    f"{path}:{number} must contain a JSON object."
                )
            result.append(value)
    return result


def prepare_panel_schedule(
    questions: Iterable[Mapping[str, Any]],
    panel_models: Sequence[Mapping[str, Any]],
    *,
    dataset_role: str,
) -> list[dict[str, Any]]:
    """Create deterministic item × model response tasks."""

    _assert_calibration_role(dataset_role)
    models = []
    for model in panel_models:
        model_id = str(model.get("model_id", "")).strip()
        tier = str(model.get("tier", "")).strip()
        model_path = str(model.get("model_path", "")).strip()
        if not (model_id and tier and model_path):
            raise DifficultyCalibrationError(
                "Each panel model needs model_id, tier, and model_path."
            )
        models.append((model_id, tier, model_path))
    if len({item[0] for item in models}) < 3:
        raise DifficultyCalibrationError(
            "The empirical panel must contain at least three distinct models."
        )
    tasks = []
    for question in sorted(questions, key=_question_id):
        question_id = _question_id(question)
        for model_id, tier, model_path in sorted(models):
            tasks.append(
                {
                    "question_id": question_id,
                    "model_id": model_id,
                    "model_tier": tier,
                    "model_path": model_path,
                    "dataset_role": dataset_role,
                    "status": "pending",
                }
            )
    return tasks


def _ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    index = 0
    while index < len(values):
        end = index + 1
        while end < len(values) and values[order[end]] == values[order[index]]:
            end += 1
        ranks[order[index:end]] = (index + end - 1) / 2.0
        index = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    if len(left) < 2 or np.std(left) == 0 or np.std(right) == 0:
        return None
    return float(np.corrcoef(left, right)[0, 1])


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(value, -30, 30)))


def fit_irt(
    matrix: np.ndarray,
    *,
    model: str = "1pl",
    iterations: int = 1200,
    learning_rate: float = 0.03,
) -> dict[str, Any]:
    """Fit a deterministic Rasch/1PL or exploratory 2PL panel model.

    Rows are panel models, columns are items, and NaN denotes a missing answer.
    Ability and item difficulty are jointly estimated by regularized gradient
    descent with mean item difficulty constrained to zero for identifiability.
    """

    if model not in {"1pl", "2pl"}:
        raise DifficultyCalibrationError("IRT model must be 1pl or 2pl.")
    observed = ~np.isnan(matrix)
    if int(observed.sum()) == 0:
        raise DifficultyCalibrationError("The panel contains no scored answers.")
    users, items = matrix.shape
    ability = np.zeros(users, dtype=float)
    difficulty = np.zeros(items, dtype=float)
    log_discrimination = np.zeros(items, dtype=float)
    for step in range(iterations):
        discrimination = (
            np.exp(np.clip(log_discrimination, -1.5, 1.5))
            if model == "2pl"
            else np.ones(items, dtype=float)
        )
        logits = discrimination[None, :] * (
            ability[:, None] - difficulty[None, :]
        )
        residual = np.where(observed, _sigmoid(logits) - matrix, 0.0)
        count_by_user = np.maximum(observed.sum(axis=1), 1)
        count_by_item = np.maximum(observed.sum(axis=0), 1)
        grad_ability = (
            (residual * discrimination[None, :]).sum(axis=1)
            / count_by_user
            + 0.01 * ability
        )
        grad_difficulty = (
            -(residual * discrimination[None, :]).sum(axis=0)
            / count_by_item
            + 0.01 * difficulty
        )
        rate = learning_rate / math.sqrt(1 + step / 200)
        ability -= rate * grad_ability
        difficulty -= rate * grad_difficulty
        difficulty -= difficulty.mean()
        ability -= ability.mean()
        if model == "2pl":
            grad_log_a = (
                (
                    residual
                    * discrimination[None, :]
                    * (ability[:, None] - difficulty[None, :])
                ).sum(axis=0)
                / count_by_item
                + 0.02 * log_discrimination
            )
            log_discrimination -= rate * grad_log_a
    return {
        "model": model,
        "ability": ability.tolist(),
        "item_difficulty": difficulty.tolist(),
        "discrimination": (
            np.exp(np.clip(log_discrimination, -1.5, 1.5)).tolist()
            if model == "2pl"
            else [1.0] * items
        ),
        "iterations": iterations,
        "learning_rate": learning_rate,
    }


def _dimension_matrix(items: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = []
    for item in items:
        profile = item.get("difficulty_profile", {})
        dimensions = profile.get("dimensions", item.get("dimensions", {}))
        row = []
        for name in DIMENSIONS:
            value = dimensions.get(name, 0.0)
            if isinstance(value, Mapping):
                value = value.get("normalized", value.get("value", 0.0))
            row.append(float(value))
        matrix.append(row)
    values = np.asarray(matrix, dtype=float)
    maximum = np.maximum(values.max(axis=0), 1.0)
    return values / maximum


def _fit_nonnegative_weights(
    features: np.ndarray,
    target: np.ndarray,
    *,
    iterations: int = 2000,
) -> np.ndarray:
    weights = np.asarray([DEFAULT_WEIGHTS[name] for name in DIMENSIONS])
    for step in range(iterations):
        prediction = features @ weights
        gradient = features.T @ (prediction - target) / max(len(target), 1)
        weights -= 0.08 / math.sqrt(1 + step / 200) * gradient
        weights = np.maximum(weights, 0.0)
        total = weights.sum()
        weights = weights / total if total else np.ones(len(weights)) / len(weights)
    return weights


def _binned_metrics(structural: np.ndarray, empirical: np.ndarray) -> list[dict[str, Any]]:
    result = []
    boundaries = ((0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01))
    for lower, upper in boundaries:
        mask = (structural >= lower) & (structural < upper)
        result.append(
            {
                "lower": lower,
                "upper": min(upper, 1.0),
                "count": int(mask.sum()),
                "mean_structural": (
                    float(structural[mask].mean()) if mask.any() else None
                ),
                "mean_empirical": (
                    float(empirical[mask].mean()) if mask.any() else None
                ),
            }
        )
    return result


def calibrate_difficulty(
    items: Sequence[Mapping[str, Any]],
    responses: Sequence[Mapping[str, Any]],
    *,
    dataset_role: str,
    dataset_hash: str,
    panel_hash: str,
) -> dict[str, Any]:
    """Estimate empirical difficulty and a frozen v2 weight candidate."""

    _assert_calibration_role(dataset_role)
    ordered_items = sorted((dict(item) for item in items), key=_question_id)
    item_ids = [_question_id(item) for item in ordered_items]
    item_index = {value: index for index, value in enumerate(item_ids)}
    model_ids = sorted({str(row["model_id"]) for row in responses})
    if len(model_ids) < 3:
        raise DifficultyCalibrationError(
            "At least three panel models are required for calibration."
        )
    model_index = {value: index for index, value in enumerate(model_ids)}
    matrix = np.full((len(model_ids), len(item_ids)), np.nan)
    tiers: dict[str, str] = {}
    for row in responses:
        question_id = _question_id(row)
        model_id = str(row["model_id"])
        if question_id not in item_index:
            raise DifficultyCalibrationError(
                f"Panel response references unknown item {question_id!r}."
            )
        matrix[model_index[model_id], item_index[question_id]] = float(
            _is_correct(row.get("is_correct"))
        )
        tiers[model_id] = str(row.get("model_tier", "unspecified"))
    if np.any(np.all(np.isnan(matrix), axis=0)):
        raise DifficultyCalibrationError(
            "Every calibration item needs at least one panel response."
        )
    accuracy = np.nanmean(matrix, axis=0)
    empirical = 1.0 - accuracy
    structural = np.asarray(
        [
            (float(item.get("difficulty", item.get("difficulty_score", 1))) - 1)
            / 9
            for item in ordered_items
        ],
        dtype=float,
    )
    features = _dimension_matrix(ordered_items)
    weights = _fit_nonnegative_weights(features, empirical)
    calibrated = np.clip(features @ weights, 0.0, 1.0)
    irt_1pl = fit_irt(matrix, model="1pl")
    irt_2pl = fit_irt(matrix, model="2pl")
    tier_consistency = {}
    for tier in sorted(set(tiers.values())):
        indices = [model_index[mid] for mid in model_ids if tiers[mid] == tier]
        tier_empirical = 1 - np.nanmean(matrix[indices, :], axis=0)
        tier_consistency[tier] = {
            "model_count": len(indices),
            "pearson_with_structural": _correlation(structural, tier_empirical),
            "spearman_with_structural": _correlation(
                _ranks(structural), _ranks(tier_empirical)
            ),
        }
    per_item = []
    for index, item_id in enumerate(item_ids):
        per_item.append(
            {
                "question_id": item_id,
                "panel_answer_count": int((~np.isnan(matrix[:, index])).sum()),
                "panel_accuracy": float(accuracy[index]),
                "empirical_difficulty": float(empirical[index]),
                "structural_difficulty": float(structural[index]),
                "calibrated_difficulty": float(calibrated[index]),
                "rasch_1pl_difficulty": irt_1pl["item_difficulty"][index],
                "irt_2pl_difficulty": irt_2pl["item_difficulty"][index],
                "irt_2pl_discrimination": irt_2pl["discrimination"][index],
            }
        )
    return {
        "schema_version": "1.0",
        "rubric_version": CALIBRATED_RUBRIC_VERSION,
        "status": "candidate",
        "created_at": utc_now(),
        "dataset_role": dataset_role,
        "dataset_sha256": dataset_hash,
        "panel_sha256": panel_hash,
        "item_count": len(item_ids),
        "model_count": len(model_ids),
        "model_ids": model_ids,
        "weights": {
            name: float(value) for name, value in zip(DIMENSIONS, weights)
        },
        "metrics": {
            "observable_pearson": _correlation(structural, empirical),
            "observable_spearman": _correlation(
                _ranks(structural), _ranks(empirical)
            ),
            "observable_mae": float(np.mean(np.abs(structural - empirical))),
            "calibrated_pearson": _correlation(calibrated, empirical),
            "calibrated_spearman": _correlation(
                _ranks(calibrated), _ranks(empirical)
            ),
            "calibrated_mae": float(np.mean(np.abs(calibrated - empirical))),
            "binned_calibration": _binned_metrics(structural, empirical),
            "model_tier_consistency": tier_consistency,
        },
        "irt": {"rasch_1pl": irt_1pl, "exploratory_2pl": irt_2pl},
        "items": per_item,
        "scientific_constraints": {
            "blind_test_used": False,
            "official_fixed_test_used": False,
            "weights_must_not_change_after_freeze": True,
        },
    }


def freeze_calibration(
    candidate: Mapping[str, Any],
    output_path: str | Path,
) -> dict[str, Any]:
    """Write an immutable-by-contract calibrated rubric artifact."""

    if candidate.get("dataset_role") not in {
        "difficulty_calibration",
        "calibration",
    }:
        raise DifficultyCalibrationError("Invalid calibration dataset role.")
    frozen = dict(candidate)
    frozen["status"] = "frozen"
    frozen["frozen_at"] = utc_now()
    frozen["content_sha256"] = canonical_sha256(
        {key: value for key, value in frozen.items() if key != "content_sha256"}
    )
    path = Path(output_path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != frozen:
            raise DifficultyCalibrationError(
                f"Refusing to overwrite frozen calibration artifact: {path}"
            )
        return existing
    atomic_json(frozen, path)
    return frozen


def calibrate_from_files(
    questions_path: str | Path,
    responses_path: str | Path,
    *,
    dataset_role: str,
) -> dict[str, Any]:
    return calibrate_difficulty(
        _load_jsonl(questions_path),
        _load_jsonl(responses_path),
        dataset_role=dataset_role,
        dataset_hash=file_sha256(questions_path),
        panel_hash=file_sha256(responses_path),
    )
