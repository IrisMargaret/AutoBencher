"""Empirical calibration for the observable mathematics difficulty rubric.

The calibration set is deliberately separate from fixed and blind evaluation
sets.  Item difficulty is estimated from a heterogeneous model response panel,
then compared with the frozen structural rubric.  No fixed-test correctness
signal is accepted by this module.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .difficulty import DEFAULT_WEIGHTS
from .experiment import atomic_json, utc_now
from .fingerprints import (
    artifact_fingerprint,
    canonical_sha256,
    file_sha256,
)


CALIBRATED_RUBRIC_VERSION = "calibrated_math_v2"
FORBIDDEN_DATASET_ROLES = {
    "blind",
    "blind_test",
    "formal_fixed_test",
    "official_fixed_test",
    "final_blind_test",
}
DIMENSIONS = tuple(DEFAULT_WEIGHTS)
DIMENSION_RAW_SCALES = {
    "reasoning_steps": 10.0,
    "operation_count": 20.0,
    "constraint_count": 10.0,
    "symbolic_depth": 10.0,
    "representation_load": 10.0,
}


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
    project_root: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Create deterministic item × model response tasks."""

    _assert_calibration_role(dataset_role)
    root = Path(project_root or Path(__file__).resolve().parents[1]).resolve()
    prompt_path = root / "prompts" / "test_taker.txt"
    prompt_sha = file_sha256(prompt_path)
    models = []
    seen_model_ids: set[str] = set()
    for model in panel_models:
        model_id = str(model.get("model_id", "")).strip()
        tier = str(model.get("tier", "")).strip()
        model_path = str(model.get("model_path", "")).strip()
        if not (model_id and tier and model_path):
            raise DifficultyCalibrationError(
                "Each panel model needs model_id, tier, and model_path."
            )
        if model_id in seen_model_ids:
            raise DifficultyCalibrationError(
                f"Duplicate panel model_id: {model_id}"
            )
        seen_model_ids.add(model_id)
        snapshot = artifact_fingerprint(
            model_path,
            project_root=root,
            allow_missing=False,
        )
        tokenizer_path = str(model.get("tokenizer_path", model_path)).strip()
        tokenizer_snapshot = artifact_fingerprint(
            tokenizer_path,
            project_root=root,
            allow_missing=False,
        )
        decoding = dict(model.get("decoding", {
            "temperature": 0.0,
            "top_p": 1.0,
            "do_sample": False,
        }))
        models.append(
            (
                model_id,
                tier,
                model_path,
                snapshot["sha256"],
                tokenizer_snapshot["sha256"],
                prompt_sha,
                canonical_sha256(decoding),
                str(model.get("provider_revision", "local_snapshot")),
                decoding,
            )
        )
    if len({item[0] for item in models}) < 3:
        raise DifficultyCalibrationError(
            "The empirical panel must contain at least three distinct models."
        )
    tasks = []
    for question in sorted(questions, key=_question_id):
        question_id = _question_id(question)
        for (
            model_id,
            tier,
            model_path,
            model_sha,
            tokenizer_sha,
            inference_prompt_sha,
            decoding_sha,
            provider_revision,
            decoding,
        ) in sorted(models):
            tasks.append(
                {
                    "question_id": question_id,
                    "model_id": model_id,
                    "model_tier": tier,
                    "model_path": model_path,
                    "model_sha256": model_sha,
                    "tokenizer_sha256": tokenizer_sha,
                    "inference_prompt_sha256": inference_prompt_sha,
                    "decoding_config": decoding,
                    "decoding_config_sha256": decoding_sha,
                    "provider_revision": provider_revision,
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
    tolerance: float = 1.0e-7,
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
    loss_history = []
    converged = False
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
        # One location constraint is sufficient. Fixing item mean to zero
        # preserves the panel's absolute average ability instead of forcing
        # both sides of the Rasch scale to zero.
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
        if step % 25 == 0 or step == iterations - 1:
            updated_a = (
                np.exp(np.clip(log_discrimination, -1.5, 1.5))
                if model == "2pl"
                else np.ones(items, dtype=float)
            )
            probability = _sigmoid(
                updated_a[None, :] * (ability[:, None] - difficulty[None, :])
            )
            likelihood = -np.where(
                observed,
                matrix * np.log(np.clip(probability, 1.0e-12, 1.0))
                + (1 - matrix)
                * np.log(np.clip(1 - probability, 1.0e-12, 1.0)),
                0.0,
            ).sum() / observed.sum()
            loss = float(
                likelihood
                + 0.005 * np.mean(ability ** 2)
                + 0.005 * np.mean(difficulty ** 2)
                + (0.01 * np.mean(log_discrimination ** 2) if model == "2pl" else 0.0)
            )
            loss_history.append({"iteration": step + 1, "loss": loss})
            if len(loss_history) >= 3 and abs(
                loss_history[-2]["loss"] - loss
            ) < tolerance:
                converged = True
                break
    return {
        "model": model,
        "ability": ability.tolist(),
        "item_difficulty": difficulty.tolist(),
        "discrimination": (
            np.exp(np.clip(log_discrimination, -1.5, 1.5)).tolist()
            if model == "2pl"
            else [1.0] * items
        ),
        "iterations": step + 1,
        "learning_rate": learning_rate,
        "converged": converged,
        "final_loss": loss_history[-1]["loss"],
        "loss_history": loss_history,
        "identification_constraint": "mean_item_difficulty_zero",
    }


def bootstrap_irt_standard_errors(
    matrix: np.ndarray,
    *,
    iterations: int = 100,
    seed: int = 2026,
) -> dict[str, Any]:
    """Cluster-bootstrap panel models for Rasch item standard errors."""
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        indices = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        fitted = fit_irt(
            matrix[indices, :],
            model="1pl",
            iterations=500,
            learning_rate=0.04,
        )
        estimates.append(fitted["item_difficulty"])
    values = np.asarray(estimates, dtype=float)
    return {
        "method": "model_cluster_bootstrap",
        "iterations": iterations,
        "item_standard_error": np.std(values, axis=0, ddof=1).tolist(),
    }


def _dimension_matrix(items: Sequence[Mapping[str, Any]]) -> np.ndarray:
    matrix = []
    for item in items:
        profile = item.get("difficulty_profile", {})
        dimensions = profile.get("dimensions", item.get("dimensions", {}))
        row = []
        for name in DIMENSIONS:
            if name not in dimensions:
                raise DifficultyCalibrationError(
                    f"Item {_question_id(item)} is missing difficulty dimension {name}"
                )
            value = dimensions[name]
            if isinstance(value, Mapping):
                if "normalized" in value:
                    numeric = float(value["normalized"])
                elif "value" in value:
                    numeric = float(value["value"]) / DIMENSION_RAW_SCALES[name]
                else:
                    raise DifficultyCalibrationError(
                        f"Dimension {name} lacks normalized or value"
                    )
            else:
                numeric = float(value) / DIMENSION_RAW_SCALES[name]
            if not math.isfinite(numeric) or not 0 <= numeric <= 1:
                raise DifficultyCalibrationError(
                    f"Dimension {name} must map to fixed [0,1], got {numeric}"
                )
            row.append(numeric)
        matrix.append(row)
    return np.asarray(matrix, dtype=float)


def _cross_validated_calibration(features, target, item_ids, folds=5):
    fold_count = min(max(2, int(folds)), len(item_ids))
    assignments = np.asarray([
        int(canonical_sha256(identifier)[:8], 16) % fold_count
        for identifier in item_ids
    ])
    # Hash assignment can leave a fold empty on small sets; stable round-robin
    # is used as a deterministic fallback.
    if len(set(assignments.tolist())) < min(fold_count, len(item_ids)):
        order = np.argsort([canonical_sha256(identifier) for identifier in item_ids])
        assignments = np.empty(len(item_ids), dtype=int)
        for rank, index in enumerate(order):
            assignments[index] = rank % fold_count
    predictions = np.zeros(len(item_ids), dtype=float)
    fold_weights = []
    for fold in range(fold_count):
        validation = assignments == fold
        training = ~validation
        if not validation.any() or not training.any():
            raise DifficultyCalibrationError("Calibration fold is empty")
        weights = _fit_nonnegative_weights(features[training], target[training])
        predictions[validation] = np.clip(features[validation] @ weights, 0.0, 1.0)
        fold_weights.append(weights.tolist())
    return predictions, {
        "method": "deterministic_item_hash_cross_validation",
        "fold_count": fold_count,
        "fold_assignments": {
            identifier: int(assignments[index])
            for index, identifier in enumerate(item_ids)
        },
        "fold_weights": fold_weights,
    }


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
    minimum_model_coverage: float = 0.80,
    minimum_models_per_item: int = 3,
    minimum_models_per_tier: int = 1,
    maximum_missing_rate: float = 0.20,
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
    seen_pairs = set()
    if not 0 < minimum_model_coverage <= 1:
        raise DifficultyCalibrationError("minimum_model_coverage must be in (0,1]")
    if minimum_models_per_item < 1 or minimum_models_per_tier < 1:
        raise DifficultyCalibrationError("Panel minimum counts must be positive")
    if not 0 <= maximum_missing_rate < 1:
        raise DifficultyCalibrationError("maximum_missing_rate must be in [0,1)")
    snapshot_by_model: dict[str, dict[str, Any]] = {}
    for row in responses:
        question_id = _question_id(row)
        model_id = str(row["model_id"])
        pair = (model_id, question_id)
        if pair in seen_pairs:
            raise DifficultyCalibrationError(
                f"Duplicate panel response for model/item: {pair}"
            )
        seen_pairs.add(pair)
        if not str(row.get("raw_response", "")).strip():
            raise DifficultyCalibrationError(
                f"Panel response lacks raw_response: {pair}"
            )
        decoding_config = row.get("decoding_config")
        if not isinstance(decoding_config, Mapping):
            raise DifficultyCalibrationError(
                f"Panel response lacks full decoding_config: {pair}"
            )
        decoding_config = dict(decoding_config)
        decoding_hash = str(row.get("decoding_config_sha256", "")).strip()
        if canonical_sha256(decoding_config) != decoding_hash:
            raise DifficultyCalibrationError(
                f"Panel response decoding config hash mismatch: {pair}"
            )
        snapshot = {
            "model_tier": str(row.get("model_tier", "")).strip(),
            "model_sha256": str(row.get("model_sha256", "")).strip(),
            "tokenizer_sha256": str(row.get("tokenizer_sha256", "")).strip(),
            "inference_prompt_sha256": str(
                row.get("inference_prompt_sha256", "")
            ).strip(),
            "decoding_config": decoding_config,
            "decoding_config_sha256": decoding_hash,
            "provider_revision": str(row.get("provider_revision", "")).strip(),
        }
        if not all(value for key, value in snapshot.items() if key != "decoding_config"):
            raise DifficultyCalibrationError(
                f"Panel response lacks immutable inference provenance: {pair}"
            )
        if model_id in snapshot_by_model and snapshot_by_model[model_id] != snapshot:
            raise DifficultyCalibrationError(
                f"Model snapshot changed within panel responses: {model_id}"
            )
        snapshot_by_model[model_id] = snapshot
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
    model_coverage = (~np.isnan(matrix)).mean(axis=1)
    item_coverage = (~np.isnan(matrix)).sum(axis=0)
    missing_rate = float(np.isnan(matrix).mean())
    if np.any(model_coverage < minimum_model_coverage):
        raise DifficultyCalibrationError(
            "Panel model coverage falls below the configured minimum"
        )
    if np.any(item_coverage < minimum_models_per_item):
        raise DifficultyCalibrationError(
            "Item model coverage falls below the configured minimum"
        )
    if missing_rate > maximum_missing_rate:
        raise DifficultyCalibrationError(
            "Panel missing-response rate exceeds the configured maximum"
        )
    tier_counts = {tier: list(tiers.values()).count(tier) for tier in set(tiers.values())}
    if len(tier_counts) < 3:
        raise DifficultyCalibrationError(
            "Calibration panel needs at least three declared ability tiers"
        )
    if min(tier_counts.values()) < minimum_models_per_tier:
        raise DifficultyCalibrationError(
            "Ability-tier model count falls below the configured minimum"
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
    calibrated_validation, cross_validation = _cross_validated_calibration(
        features, empirical, item_ids
    )
    irt_1pl = fit_irt(matrix, model="1pl")
    irt_2pl = fit_irt(matrix, model="2pl")
    irt_bootstrap = bootstrap_irt_standard_errors(matrix)
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
                "calibrated_validation_difficulty": float(
                    calibrated_validation[index]
                ),
                "rasch_1pl_difficulty": irt_1pl["item_difficulty"][index],
                "rasch_1pl_standard_error": irt_bootstrap[
                    "item_standard_error"
                ][index],
                "empirical_difficulty_standard_error": float(
                    math.sqrt(
                        accuracy[index]
                        * (1 - accuracy[index])
                        / max((~np.isnan(matrix[:, index])).sum(), 1)
                    )
                ),
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
        "model_snapshots": {
            model_id: snapshot_by_model[model_id]
            for model_id in model_ids
        },
        "panel_coverage": {
            "model_coverage": {
                model_id: float(model_coverage[model_index[model_id]])
                for model_id in model_ids
            },
            "minimum_models_per_item": int(item_coverage.min()),
            "missing_rate": missing_rate,
            "tier_model_counts": dict(sorted(tier_counts.items())),
            "acceptance_thresholds": {
                "minimum_model_coverage": minimum_model_coverage,
                "minimum_models_per_item": minimum_models_per_item,
                "minimum_models_per_tier": minimum_models_per_tier,
                "maximum_missing_rate": maximum_missing_rate,
            },
        },
        "raw_response_records": [dict(row) for row in responses],
        "weights": {
            name: float(value) for name, value in zip(DIMENSIONS, weights)
        },
        "metrics": {
            "observable_pearson": _correlation(structural, empirical),
            "observable_spearman": _correlation(
                _ranks(structural), _ranks(empirical)
            ),
            "observable_mae": float(np.mean(np.abs(structural - empirical))),
            "calibrated_pearson": _correlation(calibrated_validation, empirical),
            "calibrated_spearman": _correlation(
                _ranks(calibrated_validation), _ranks(empirical)
            ),
            "calibrated_mae": float(
                np.mean(np.abs(calibrated_validation - empirical))
            ),
            "calibrated_metric_scope": "held_out_cross_validation",
            "calibrated_in_sample_pearson_descriptive_only": _correlation(
                calibrated, empirical
            ),
            "calibrated_in_sample_mae_descriptive_only": float(
                np.mean(np.abs(calibrated - empirical))
            ),
            "cross_validation": cross_validation,
            "binned_calibration": _binned_metrics(structural, empirical),
            "model_tier_consistency": tier_consistency,
        },
        "irt": {
            "rasch_1pl": irt_1pl,
            "rasch_1pl_bootstrap": irt_bootstrap,
            "exploratory_2pl": {
                **irt_2pl,
                "interpretation": (
                    "exploratory_only; discrimination is weakly identified "
                    "when the model panel is small"
                ),
            },
        },
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
    minimum_model_coverage: float = 0.80,
    minimum_models_per_item: int = 3,
    minimum_models_per_tier: int = 1,
    maximum_missing_rate: float = 0.20,
) -> dict[str, Any]:
    return calibrate_difficulty(
        _load_jsonl(questions_path),
        _load_jsonl(responses_path),
        dataset_role=dataset_role,
        dataset_hash=file_sha256(questions_path),
        panel_hash=file_sha256(responses_path),
        minimum_model_coverage=minimum_model_coverage,
        minimum_models_per_item=minimum_models_per_item,
        minimum_models_per_tier=minimum_models_per_tier,
        maximum_missing_rate=maximum_missing_rate,
    )
