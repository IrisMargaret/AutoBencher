import hashlib
import json
from pathlib import Path

import pytest
import yaml

from autobencher.evaluation_sets import (
    file_sha256,
    resolve_active_evaluation_set,
)
from autobencher.config import load_project_config
from autobencher.evaluation_audit import audit_evaluation_questions
from autobencher.fixed_benchmark import load_fixed_test_set
from autobencher.training_protocol import split_by_template_cluster
from autobencher.truth_solver import TruthSolver


def _registry(tmp_path):
    registry = {
        "schema_version": "1.0",
        "sets": [
            {
                "id": "dev",
                "role": "development_regression",
                "version": "1",
                "dataset_path": "dev.json",
                "minimum_questions_per_subcategory": 1,
                "permissions": {"training": False},
            },
            {
                "id": "blind",
                "role": "blind_final",
                "version": "1",
                "dataset_path_env": "BLIND_PATH",
                "dataset_sha256_env": "BLIND_HASH",
                "release_token_env": "BLIND_TOKEN",
                "minimum_questions_per_subcategory": 1,
                "permissions": {"training": False},
            },
        ],
    }
    path = tmp_path / "registry.yaml"
    path.write_text(yaml.safe_dump(registry), encoding="utf-8")
    return path


def test_development_set_resolves_from_registry(tmp_path):
    registry = _registry(tmp_path)
    config = {
        "evaluation_sets": {
            "registry_path": str(registry),
            "active_set": "dev",
            "phase": "development",
            "allow_blind_evaluation": False,
        },
        "fixed_test": {"dataset_path": "ignored.json"},
        "finetune": {"enabled": True},
    }
    selected = resolve_active_evaluation_set(config, tmp_path)
    assert selected["role"] == "development_regression"
    assert selected["dataset_path"] == (tmp_path / "dev.json").as_posix()


def test_blind_set_fails_closed_and_uses_only_environment_path(tmp_path):
    registry = _registry(tmp_path)
    dataset = tmp_path / "blind.json"
    dataset.write_text(json.dumps({"questions": []}), encoding="utf-8")
    config = {
        "evaluation_sets": {
            "registry_path": str(registry),
            "active_set": "blind",
            "phase": "development",
            "allow_blind_evaluation": False,
        },
        "fixed_test": {"dataset_path": "must-not-be-used.json"},
        "finetune": {"enabled": False},
    }
    env = {
        "BLIND_PATH": str(dataset),
        "BLIND_HASH": file_sha256(dataset),
        "BLIND_TOKEN": "released",
        "AUTOBENCHER_BLIND_RELEASE_TOKEN_SHA256": hashlib.sha256(
            b"released"
        ).hexdigest(),
    }
    with pytest.raises(PermissionError):
        resolve_active_evaluation_set(config, tmp_path, environment=env)
    config["evaluation_sets"].update(
        {"phase": "blind_evaluation", "allow_blind_evaluation": True}
    )
    selected = resolve_active_evaluation_set(
        config,
        tmp_path,
        environment=env,
    )
    assert selected["dataset_path"] == dataset.as_posix()
    assert selected["released_sha256"] == file_sha256(dataset)


def test_template_cluster_split_is_deterministic_and_has_zero_overlap():
    records = []
    for template_index in range(12):
        for value in (1, 2):
            records.append(
                {
                    "instruction": "solve",
                    "input": f"Template {template_index}: compute {value} + 7",
                    "output": "{}",
                    "_metadata": {
                        "template_signature": f"template-{template_index}",
                        "category": "Arithmetic",
                    },
                }
            )
    config = {
        "train_fraction": 0.8,
        "validation_fraction": 0.1,
        "internal_test_fraction": 0.1,
    }
    first, first_manifest = split_by_template_cluster(
        records,
        config,
        seed=42,
    )
    second, second_manifest = split_by_template_cluster(
        records,
        config,
        seed=42,
    )
    assert first == second
    assert first_manifest["cluster_assignments_sha256"] == second_manifest[
        "cluster_assignments_sha256"
    ]
    assert first_manifest["template_overlap_count"] == 0
    assert all(first[name] for name in ("train", "validation", "internal_test"))


def test_all_81_development_questions_have_independent_recomputation():
    root = Path(__file__).resolve().parents[1]
    config, _ = load_project_config(
        root / "configs" / "math_flywheel.yaml",
        validate_paths=False,
    )
    questions, _ = load_fixed_test_set(config, root)
    audit = audit_evaluation_questions(
        questions,
        normalization_config=config,
        solver=TruthSolver.from_config(config),
    )
    assert audit["question_count"] == 81
    assert audit["all_independently_verified"] is True
    assert audit["status_counts"] == {"independently_verified": 81}


def test_retention_holdout_has_paper_sized_dimension_coverage():
    root = Path(__file__).resolve().parents[1]
    questions, metadata = load_fixed_test_set(
        {
            "fixed_test": {
                "dataset_path": "benchmarks/retention_regression_set.json",
                "require_all_subcategories": False,
            }
        },
        root,
    )
    dimensions = {}
    for question in questions:
        dimension = question["retention_dimension"]
        dimensions[dimension] = dimensions.get(dimension, 0) + 1
        assert question["source_dataset"] == "project_native"
        assert question["verification"]["training_use_prohibited"] is True
    assert metadata["question_count"] == 120
    assert len(dimensions) == 6
    assert set(dimensions.values()) == {20}
