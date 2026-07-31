import hashlib
import json
from pathlib import Path

import pytest
import yaml

from build_project_fixed_benchmark import build_payload
from autobencher.fixed_benchmark import (
    PROJECT_FIXED_TEST_SET,
    install_project_fixed_test_set,
    load_fixed_test_set,
)


def test_project_fixed_installer_is_offline_idempotent_and_complete(tmp_path):
    root = tmp_path / "data"
    output = root / "benchmarks" / "fixed_math_test_set_v2.json"

    first = install_project_fixed_test_set(
        output,
        allowed_data_root=root,
    )
    second = install_project_fixed_test_set(
        output,
        allowed_data_root=root,
    )

    assert first["status"] == "installed"
    assert second["status"] == "already_installed"
    assert first["network_access"] is False
    assert output.read_bytes() == PROJECT_FIXED_TEST_SET.read_bytes()
    assert first["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert first["covered_subcategory_count"] == 27
    assert first["question_count"] == 81

    manifest = json.loads(
        output.with_suffix(".json.manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["question_source"] == "project_native"
    assert "openai/gsm8k" in manifest["excluded_question_sources"]


def test_project_fixed_installer_rejects_output_outside_data_root(tmp_path):
    with pytest.raises(ValueError, match="beneath allowed data root"):
        install_project_fixed_test_set(
            tmp_path / "outside.json",
            allowed_data_root=tmp_path / "data",
        )


def test_all_active_configs_exclude_external_fixed_question_suite():
    project_root = Path(__file__).parents[1]
    config_paths = [
        project_root / "configs" / "math_flywheel.yaml",
        project_root / "configs" / "math_flywheel_local.yaml",
        project_root / "configs" / "math_flywheel_volcengine.yaml",
        *sorted((project_root / "configs" / "environments").glob("*.yaml")),
    ]
    for path in config_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        fixed_test = payload.get("fixed_test", {})
        if "dataset_path" not in fixed_test:
            continue
        assert Path(fixed_test["dataset_path"]).name in {
            "fixed_math_test_set.json",
            "fixed_math_test_set_v2.json",
        }, path
        assert fixed_test.get("require_all_subcategories") is True, path


def test_checked_in_fixed_questions_are_project_native():
    config = {
        "fixed_test": {
            "dataset_path": PROJECT_FIXED_TEST_SET.as_posix(),
            "require_all_subcategories": True,
        }
    }
    questions, metadata = load_fixed_test_set(config)
    assert len(questions) == 81
    assert metadata["source_counts"] == {"project_native": 81}


def test_checked_in_benchmark_is_reproducible_and_balanced():
    checked_in = json.loads(
        PROJECT_FIXED_TEST_SET.read_text(encoding="utf-8")
    )
    rebuilt = build_payload(PROJECT_FIXED_TEST_SET)
    assert rebuilt == checked_in

    counts = {}
    for question in checked_in["questions"]:
        key = (question["category"], question["sub_category"])
        counts[key] = counts.get(key, 0) + 1
        assert question["source_dataset"] == "project_native"
        assert question["verification"]["canonical_parse_passed"] is True
    assert set(counts.values()) == {3}
    assert all(
        reference["questions_copied"] is False
        for reference in checked_in["design_references"]
    )
