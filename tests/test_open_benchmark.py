from __future__ import annotations

import json
from pathlib import Path

import yaml
import pytest

from autobencher import open_benchmark
from autobencher.config import load_resolved_config
from autobencher.fixed_benchmark import (
    fixed_benchmark_summary,
    load_fixed_test_set,
)


class FakeDataset(list):
    def __init__(self, records, fingerprint):
        super().__init__(records)
        self._fingerprint = fingerprint


@pytest.fixture
def config():
    path = (
        Path(__file__).parents[1]
        / "configs"
        / "math_flywheel_smoke_test.yaml"
    )
    return load_resolved_config(path)[0]


def test_open_suite_uses_only_test_splits_and_records_provenance(
    tmp_path,
    monkeypatch,
):
    manifest = {
        "name": "fixture-open-suite",
        "sources": [
            {
                "type": "gsm8k",
                "split": "test",
                "samples": 1,
                "seed": 1,
            },
            {
                "type": "math",
                "split": "test",
                "configs": ["algebra"],
                "samples_per_config": 1,
                "seed": 2,
            },
            {
                "type": "mmlu_math",
                "split": "test",
                "configs": ["elementary_mathematics"],
                "samples_per_config": 1,
                "seed": 3,
            },
        ],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(
        yaml.safe_dump(manifest),
        encoding="utf-8",
    )
    requested = []

    def fake_load(dataset_name, config_name, split, **kwargs):
        del kwargs
        requested.append((dataset_name, config_name, split))
        if dataset_name == "openai/gsm8k":
            return FakeDataset(
                [
                    {
                        "question": "A car travels 60 miles in 2 hours. What is its speed?",
                        "answer": "60 / 2 = 30. #### 30",
                    }
                ],
                "gsm-fingerprint",
            )
        if dataset_name == "EleutherAI/hendrycks_math":
            return FakeDataset(
                [
                    {
                        "problem": "Compute 2+3.",
                        "solution": "We obtain \\boxed{5}.",
                        "level": "Level 1",
                    }
                ],
                "math-fingerprint",
            )
        return FakeDataset(
            [
                {
                    "question": "What is 2+2?",
                    "choices": ["3", "4", "5", "6"],
                    "answer": 1,
                }
            ],
            "mmlu-fingerprint",
        )

    monkeypatch.setattr(open_benchmark, "_load_hf_dataset", fake_load)
    output = tmp_path / "data" / "open_suite.json"
    result = open_benchmark.build_open_fixed_suite(
        manifest_path,
        output,
        tmp_path / "cache",
        allowed_data_root=tmp_path,
    )
    assert result["question_count"] == 3
    assert all(split == "test" for _, _, split in requested)
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["selection_policy"]["training_splits_forbidden"] is True
    assert {item["source_split"] for item in payload["questions"]} == {"test"}
    assert {
        item["source_dataset"] for item in payload["questions"]
    } == {
        "openai/gsm8k",
        "EleutherAI/hendrycks_math",
        "cais/mmlu",
    }
    mmlu = next(
        item
        for item in payload["questions"]
        if item["source_dataset"] == "cais/mmlu"
    )
    assert mmlu["answer_type"] == "multiple_choice"
    assert mmlu["canonical_answer"] == "B"
    assert "A. 3" in mmlu["question"]
    repeated = open_benchmark.build_open_fixed_suite(
        manifest_path,
        output,
        tmp_path / "cache",
        allowed_data_root=tmp_path,
    )
    assert repeated["status"] == "already_exists"
    assert len(requested) == 3


def test_open_suite_refuses_manifest_drift_for_existing_artifact(tmp_path):
    output = tmp_path / "data" / "fixed.json"
    output.parent.mkdir(parents=True)
    output.write_text(
        json.dumps(
            {
                "builder_manifest_sha256": "old",
                "question_count": 0,
                "source_counts": {},
                "sources": [],
            }
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("name: new\nsources: []\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="different manifest"):
        open_benchmark.build_open_fixed_suite(
            manifest,
            output,
            tmp_path / "cache",
            allowed_data_root=tmp_path,
        )


def test_open_suite_refuses_silently_shrunken_source(
    tmp_path,
    monkeypatch,
):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "sources": [
                    {
                        "type": "gsm8k",
                        "split": "test",
                        "samples": 2,
                        "seed": 1,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        open_benchmark,
        "_load_hf_dataset",
        lambda *args, **kwargs: FakeDataset(
            [
                {
                    "question": "Only one question.",
                    "answer": "#### 1",
                }
            ],
            "small",
        ),
    )
    with pytest.raises(RuntimeError, match="expected exactly 2"):
        open_benchmark.build_open_fixed_suite(
            manifest,
            tmp_path / "data" / "fixed.json",
            tmp_path / "cache",
            allowed_data_root=tmp_path,
        )


def test_open_suite_refuses_output_outside_allowed_root(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("name: empty\nsources: []\n", encoding="utf-8")
    outside = tmp_path.parent / "outside-open-suite.json"
    try:
        open_benchmark.build_open_fixed_suite(
            manifest,
            outside,
            tmp_path / "cache",
            allowed_data_root=tmp_path,
        )
    except RuntimeError as exc:
        assert "allowed data root" in str(exc)
    else:
        raise AssertionError("Expected an allowed-root failure")


def test_open_suite_rejects_training_split(tmp_path, monkeypatch):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "leaky-suite",
                "sources": [
                    {
                        "type": "gsm8k",
                        "split": "train",
                        "samples": 1,
                        "seed": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        open_benchmark,
        "_load_hf_dataset",
        lambda *args, **kwargs: pytest.fail(
            "A forbidden split must fail before downloading data"
        ),
    )
    with pytest.raises(ValueError, match="must use split='test'"):
        open_benchmark.build_open_fixed_suite(
            manifest,
            tmp_path / "data" / "fixed.json",
            tmp_path / "cache",
            allowed_data_root=tmp_path,
        )


def test_deepmind_local_source_must_stay_under_data_root(tmp_path):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "outside-local-source",
                "sources": [
                    {
                        "type": "deepmind_mathematics_local",
                        "local_root": str(tmp_path.parent),
                        "samples": 1,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="local_root must remain"):
        open_benchmark.build_open_fixed_suite(
            manifest,
            tmp_path / "data" / "fixed.json",
            tmp_path / "cache",
            allowed_data_root=tmp_path,
        )


def test_fixed_loader_and_summary_keep_source_provenance(
    tmp_path,
    config,
):
    payload = json.loads(
        (Path(__file__).parents[1] / "benchmarks" / "fixed_math_test_set.json")
        .read_text(encoding="utf-8")
    )
    payload["questions"][0]["source_dataset"] = "openai/gsm8k"
    path = tmp_path / "fixed.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    config["fixed_test"]["dataset_path"] = str(path)
    questions, metadata = load_fixed_test_set(config)
    assert metadata["source_counts"]["openai/gsm8k"] == 1
    assert "builder_manifest_sha256" in metadata
    records = [
        {
            **record,
            "is_correct": index == 0,
            "parse_status": "success",
        }
        for index, record in enumerate(questions)
    ]
    summary = fixed_benchmark_summary(
        records,
        stage="test",
        model_name="fixture",
        dataset_sha256=metadata["sha256"],
    )
    source = {
        item["source_dataset"]: item
        for item in summary["source_statistics"]
    }
    assert source["openai/gsm8k"]["accuracy"] == 1.0
