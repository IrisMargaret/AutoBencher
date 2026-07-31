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
    # Seed every split first. Put smaller clusters into the two holdouts to
    # avoid a large template family dominating a small validation set.
    bootstrap = ("train", "validation", "internal_test")
    for index, (key, members) in enumerate(clusters):
        if index < len(bootstrap):
            split = bootstrap[index]
        else:
            split = max(
                SPLIT_NAMES,
                key=lambda name: (
                    target[name] - len(result[name]),
                    -len(result[name]),
                    name,
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
        "cluster_assignments_sha256": hashlib.sha256(
            json.dumps(
                cluster_assignments,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
    }
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
    root = Path(output_dir).expanduser().resolve()
    paths = {}
    for name in SPLIT_NAMES:
        path = root / f"dataset_{name}.jsonl"
        _write_alpaca(splits[name], path)
        paths[name] = path.as_posix()
    manifest["paths"] = paths
    atomic_json(manifest, root / "split_manifest.json")
    return manifest
