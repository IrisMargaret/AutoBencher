"""Leakage-safe template-cluster splits for internal model selection."""

from __future__ import annotations

import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from .dataset import template_signature
from .experiment import atomic_json
from .budget_ledger import training_record_tokens


SPLIT_NAMES = ("train", "validation", "internal_test")
ALPACA_FIELDS = ("instruction", "input", "output")


def _cluster_key(record: Mapping[str, Any]) -> str:
    metadata = record.get("_metadata")
    if isinstance(metadata, Mapping):
        existing = str(metadata.get("template_signature", "")).strip()
        if existing:
            return existing
    return template_signature(record.get("input", ""))


def _stable_rank(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode("utf-8")).hexdigest()


def _normalize_fractions(config: Mapping[str, Any]) -> dict[str, float]:
    values = {
        "train": float(config["train_fraction"]),
        "validation": float(config["validation_fraction"]),
        "internal_test": float(config["internal_test_fraction"]),
    }
    if any(value <= 0 for value in values.values()):
        raise ValueError("All training split fractions must be positive")
    total = sum(values.values())
    if abs(total - 1.0) > 1.0e-9:
        raise ValueError("Training split fractions must sum to 1.0")
    return values


def split_by_template_cluster(
    records: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Assign whole template clusters with deterministic, balanced greediness."""
    normalized = [dict(record) for record in records]
    if not normalized:
        raise ValueError("Cannot split an empty training dataset")
    fractions = _normalize_fractions(config)
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in normalized:
        key = _cluster_key(record)
        if not key:
            raise ValueError("Training record has an empty template signature")
        grouped[key].append(record)
    if len(grouped) < 3:
        raise ValueError(
            "Template-cluster splitting needs at least three distinct clusters "
            "so train, validation, and internal_test are all non-empty"
        )

    target = {
        name: fractions[name] * len(normalized)
        for name in SPLIT_NAMES
    }
    result: dict[str, list[dict[str, Any]]] = {
        name: [] for name in SPLIT_NAMES
    }
    cluster_assignments: dict[str, str] = {}
    clusters = sorted(
        grouped.items(),
        key=lambda item: (
            -len(item[1]),
            _stable_rank(seed, item[0]),
        ),
    )
    global_categories = Counter(
        str((record.get("_metadata") or {}).get("category", "unknown"))
        for record in normalized
    )
    global_sources = Counter(
        str((record.get("_metadata") or {}).get("training_source", "unknown"))
        for record in normalized
    )
    global_correctness = Counter(
        "correct"
        if str((record.get("_metadata") or {}).get("training_source"))
        == "correct_retention_samples"
        else "incorrect"
        for record in normalized
    )

    def assignment_cost(split: str, members: list[dict[str, Any]]) -> float:
        projected_counts = {
            name: len(result[name]) + (len(members) if name == split else 0)
            for name in SPLIT_NAMES
        }
        count_cost = sum(
            ((projected_counts[name] - target[name]) / max(target[name], 1.0)) ** 2
            for name in SPLIT_NAMES
        )
        balance_cost = 0.0
        for feature_counts, getter in (
            (
                global_categories,
                lambda record: str(
                    (record.get("_metadata") or {}).get("category", "unknown")
                ),
            ),
            (
                global_sources,
                lambda record: str(
                    (record.get("_metadata") or {}).get(
                        "training_source", "unknown"
                    )
                ),
            ),
            (
                global_correctness,
                lambda record: (
                    "correct"
                    if str((record.get("_metadata") or {}).get("training_source"))
                    == "correct_retention_samples"
                    else "incorrect"
                ),
            ),
        ):
            current = Counter(getter(record) for record in result[split])
            current.update(getter(record) for record in members)
            balance_cost += sum(
                (
                    (current[label] - fractions[split] * total)
                    / max(fractions[split] * total, 1.0)
                ) ** 2
                for label, total in feature_counts.items()
            )
        # Count fidelity is primary; stratification resolves near ties.
        return count_cost + 0.04 * balance_cost

    for index, (key, members) in enumerate(clusters):
        remaining = len(clusters) - index
        empty = [name for name in SPLIT_NAMES if not result[name]]
        candidates = empty if remaining == len(empty) else list(SPLIT_NAMES)
        split = min(
            candidates,
            key=lambda name: (
                assignment_cost(name, members),
                _stable_rank(seed, f"{key}|{name}"),
            ),
        )
        result[split].extend(members)
        cluster_assignments[key] = split

    split_clusters = {
        name: {
            key for key, split in cluster_assignments.items() if split == name
        }
        for name in SPLIT_NAMES
    }
    overlaps = {
        f"{left}__{right}": sorted(split_clusters[left] & split_clusters[right])
        for index, left in enumerate(SPLIT_NAMES)
        for right in SPLIT_NAMES[index + 1 :]
    }
    if any(overlaps.values()):
        raise AssertionError("Template cluster overlap detected")

    manifest = {
        "schema_version": "1.0",
        "strategy": "template_cluster",
        "seed": int(seed),
        "fractions": fractions,
        "record_count": len(normalized),
        "cluster_count": len(grouped),
        "split_record_counts": {
            name: len(result[name]) for name in SPLIT_NAMES
        },
        "split_fractions": {
            name: len(result[name]) / len(normalized) for name in SPLIT_NAMES
        },
        "fraction_deviation": {
            name: len(result[name]) / len(normalized) - fractions[name]
            for name in SPLIT_NAMES
        },
        "split_cluster_counts": {
            name: len(split_clusters[name]) for name in SPLIT_NAMES
        },
        "template_overlap": overlaps,
        "template_overlap_count": sum(len(value) for value in overlaps.values()),
        "category_counts": {
            name: dict(
                sorted(
                    Counter(
                        str((record.get("_metadata") or {}).get("category"))
                        for record in result[name]
                    ).items()
                )
            )
            for name in SPLIT_NAMES
        },
        "missing_categories": {
            name: sorted(set(global_categories) - {
                str((record.get("_metadata") or {}).get("category", "unknown"))
                for record in result[name]
            })
            for name in SPLIT_NAMES
        },
        "training_source_counts": {
            name: dict(sorted(Counter(
                str((record.get("_metadata") or {}).get("training_source", "unknown"))
                for record in result[name]
            ).items()))
            for name in SPLIT_NAMES
        },
        "correctness_counts": {
            name: dict(sorted(Counter(
                "correct"
                if str((record.get("_metadata") or {}).get("training_source"))
                == "correct_retention_samples"
                else "incorrect"
                for record in result[name]
            ).items()))
            for name in SPLIT_NAMES
        },
        "correct_fraction": {
            name: (
                sum(
                    str((record.get("_metadata") or {}).get("training_source"))
                    == "correct_retention_samples"
                    for record in result[name]
                ) / len(result[name])
            )
            for name in SPLIT_NAMES
        },
        "warnings": [],
        "cluster_assignments_sha256": hashlib.sha256(
            json.dumps(
                cluster_assignments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
    largest_cluster = max(len(members) for members in grouped.values())
    if largest_cluster > max(target.values()):
        manifest["warnings"].append(
            {
                "type": "oversized_template_cluster",
                "cluster_size": largest_cluster,
                "largest_split_target": max(target.values()),
                "message": "Exact requested split fractions are infeasible.",
            }
        )
    if any(abs(value) > largest_cluster / len(normalized) for value in manifest["fraction_deviation"].values()):
        manifest["warnings"].append(
            {
                "type": "split_fraction_deviation",
                "message": "Observed deviation exceeds one largest-cluster fraction.",
            }
        )
    return result, manifest


def _write_alpaca(records: Iterable[Mapping[str, Any]], path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    count = 0
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            payload = {key: str(record[key]) for key in ALPACA_FIELDS}
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    return count


def write_training_splits(
    records: Iterable[Mapping[str, Any]],
    output_dir: str | Path,
    config: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    splits, manifest = split_by_template_cluster(records, config, seed=seed)
    return write_precomputed_training_splits(splits, output_dir, manifest)


def write_precomputed_training_splits(
    splits: Mapping[str, Iterable[Mapping[str, Any]]],
    output_dir: str | Path,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Write an already audited split without recomputing its assignment."""
    manifest = dict(manifest)
    materialized = {
        name: [dict(record) for record in splits[name]] for name in SPLIT_NAMES
    }
    total = sum(len(materialized[name]) for name in SPLIT_NAMES)
    configured_fractions = dict(manifest.get("fractions", {}))
    all_categories = {
        str((record.get("_metadata") or {}).get("category", "unknown"))
        for records in materialized.values()
        for record in records
    }
    manifest["record_count"] = total
    manifest["split_record_counts"] = {
        name: len(materialized[name]) for name in SPLIT_NAMES
    }
    manifest["split_fractions"] = {
        name: len(materialized[name]) / total for name in SPLIT_NAMES
    }
    manifest["fraction_deviation"] = {
        name: manifest["split_fractions"][name]
        - float(configured_fractions.get(name, 0.0))
        for name in SPLIT_NAMES
    }
    manifest["category_counts"] = {
        name: dict(sorted(Counter(
            str((record.get("_metadata") or {}).get("category", "unknown"))
            for record in materialized[name]
        ).items()))
        for name in SPLIT_NAMES
    }
    manifest["missing_categories"] = {
        name: sorted(all_categories - set(manifest["category_counts"][name]))
        for name in SPLIT_NAMES
    }
    manifest["training_source_counts"] = {
        name: dict(sorted(Counter(
            str((record.get("_metadata") or {}).get("training_source", "unknown"))
            for record in materialized[name]
        ).items()))
        for name in SPLIT_NAMES
    }
    manifest["correctness_counts"] = {
        name: dict(sorted(Counter(
            "correct"
            if str((record.get("_metadata") or {}).get("training_source"))
            == "correct_retention_samples"
            else "incorrect"
            for record in materialized[name]
        ).items()))
        for name in SPLIT_NAMES
    }
    manifest["correct_fraction"] = {
        name: (
            manifest["correctness_counts"][name].get("correct", 0)
            / len(materialized[name])
        )
        for name in SPLIT_NAMES
    }
    root = Path(output_dir).expanduser().resolve()
    paths = {}
    for name in SPLIT_NAMES:
        path = root / f"dataset_{name}.jsonl"
        _write_alpaca(materialized[name], path)
        paths[name] = path.as_posix()
    manifest["paths"] = paths
    manifest["split_token_counts"] = {
        name: training_record_tokens(materialized[name]) for name in SPLIT_NAMES
    }
    atomic_json(manifest, root / "split_manifest.json")
    return manifest
