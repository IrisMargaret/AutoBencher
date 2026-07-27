import json
import os
import sys
from pathlib import Path

import pytest

from autobencher.attribution_eval import evaluate_review_csv, export_review_sample
from autobencher.config import load_resolved_config
from autobencher.dataset import (
    build_training_dataset,
    exact_signature,
    tfidf_cosine,
    token_jaccard,
    write_alpaca_jsonl,
)
from autobencher.experiment import ProgressManager, ResearchRun, atomic_json
from tool_util import dump_standard_json, manage_hard_pool


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "math_flywheel_smoke_test.yaml"


@pytest.fixture
def config():
    return load_resolved_config(CONFIG_PATH)[0]


def record(question, answer, *, correct=False, accuracy=0.2, **extra):
    return {
        "question": question,
        "gold_answer": answer,
        "canonical_answer": answer,
        "category": "Arithmetic",
        "sub_category": "Integer Operations",
        "answer_type": "integer",
        "is_correct": correct,
        "sub_category_accuracy": accuracy,
        "evaluator_confidence": 0.95,
        "question_parse_success": True,
        "answer_validation_success": True,
        **extra,
    }


def test_exact_signature_is_stable():
    assert exact_signature(record("What is 2 + 2?", "4")) == exact_signature(
        record(" what  is 2 + 2? ", "4")
    )


def test_similarity_metrics_detect_close_text():
    left = "Solve the integer equation x plus 2 equals 5"
    right = "Solve the integer equation x plus 2 equals 6"
    assert token_jaccard(left, right) > 0.5
    assert tfidf_cosine(left, right) > 0.7


def test_dataset_exact_dedup_and_noise_filter(config):
    records = [
        record("What is 2 + 2?", "4"),
        record("What is 2 + 2?", "4"),
        record("What is 3 + 3?", "6", tool_violation=True),
    ]
    selected, manifest, rejected = build_training_dataset(records, config)
    assert len(selected) == 1
    assert manifest["rejection_reasons"]["exact_duplicate"] == 1
    assert manifest["rejection_reasons"]["tool_violation"] == 1
    assert len(rejected) == 2


def test_dataset_template_cluster_limit(config):
    config["dataset"]["near_duplicate_threshold"] = 1.0
    config["dataset"]["semantic_dedup"] = False
    config["dataset"]["max_samples_per_template_cluster"] = 1
    records = [
        record("Compute 21 plus 4.", "25"),
        record("Compute 37 plus 9.", "46"),
    ]
    selected, manifest, _ = build_training_dataset(records, config)
    assert len(selected) == 1
    assert manifest["rejection_reasons"]["template_duplicate"] == 1


def test_dataset_contains_correct_and_incorrect_sources(config):
    config["dataset"]["near_duplicate_threshold"] = 1.0
    config["dataset"]["semantic_dedup"] = False
    records = [
        record(
            f"Boundary problem number {index} asks for {index} plus one.",
            str(index + 1),
        )
        for index in range(5)
    ] + [
        record(
            f"Retention problem number {index} asks for {index} times two.",
            str(index * 2),
            correct=True,
        )
        for index in range(5, 10)
    ]
    _, manifest, _ = build_training_dataset(records, config)
    assert manifest["selected_mix_counts"]["incorrect_boundary_samples"] > 0
    assert manifest["selected_mix_counts"]["correct_retention_samples"] > 0


def test_alpaca_writer_omits_private_metadata(tmp_path):
    output = tmp_path / "dataset.jsonl"
    count = write_alpaca_jsonl(
        [
            {
                "instruction": "Solve.",
                "input": "1+1",
                "output": "2",
                "_metadata": {"private": True},
            }
        ],
        output,
    )
    assert count == 1
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload == {"instruction": "Solve.", "input": "1+1", "output": "2"}
    assert not output.with_name(output.name + ".tmp").exists()


def test_atomic_json_is_pretty_and_complete(tmp_path):
    output = tmp_path / "artifact.json"
    atomic_json({"alpha": 1, "items": [1, 2]}, output)
    text = output.read_text(encoding="utf-8")
    assert text.endswith("\n")
    assert "\n  \"alpha\"" in text
    assert not output.with_name(output.name + ".tmp").exists()


def test_progress_manager_rejects_nested_stages(config):
    manager = ProgressManager(config)
    with manager.stage("outer", 1):
        with pytest.raises(RuntimeError, match="still active"):
            with manager.stage("inner", 1):
                pass
    assert manager._active is None


def test_progress_manager_disables_library_progress(config):
    ProgressManager(config)
    assert os.environ["HF_DATASETS_DISABLE_PROGRESS_BARS"] == "1"
    assert os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] == "1"


def test_progress_manager_creates_one_dynamic_bar_per_stage(
    config,
    monkeypatch,
):
    config["logging"]["progress_enabled"] = True
    created = []

    class FakeStderr:
        @staticmethod
        def isatty():
            return True

    class FakeProgress:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.updated = 0
            self.closed = False
            created.append(self)

        def update(self, amount=1):
            self.updated += amount

        def close(self):
            self.closed = True

    monkeypatch.setattr(sys, "stderr", FakeStderr())
    monkeypatch.setattr("tqdm.auto.tqdm", FakeProgress)
    manager = ProgressManager(config)
    with manager.stage("Generate", 3, cycle=1, iteration=1) as progress:
        progress.update(3)
    assert len(created) == 1
    assert created[0].updated == 3
    assert created[0].closed is True
    assert created[0].kwargs["position"] == 0
    assert created[0].kwargs["leave"] is False


def test_hard_pool_resume_is_idempotent(tmp_path):
    inference = tmp_path / "inference.json"
    hard_pool = tmp_path / "hard_pool.json"
    dump_standard_json(
        [
            {
                **record("What is 2 + 3?", "5"),
                "test_taker_response": "6",
                "is_correct": False,
                "difficulty": 2,
            }
        ],
        inference,
    )
    manage_hard_pool(inference, hard_pool, source_iter=1, source_cycle=1)
    manage_hard_pool(inference, hard_pool, source_iter=1, source_cycle=1)
    pool = json.loads(hard_pool.read_text(encoding="utf-8"))
    assert pool[0]["occurrences"] == 1


def test_research_run_writes_reproducibility_snapshot(tmp_path, config):
    config = {**config, "paths": {**config["paths"], "output_root": str(tmp_path)}}
    provenance = {"config_hash": "abc123", "sources": {}, "schema_version": "1.0"}
    run = ResearchRun(config, provenance, "mock-run", ROOT)
    run.initialize({"config": str(CONFIG_PATH)})
    assert (run.run_dir / "resolved_config.json").is_file()
    assert (run.run_dir / "environment.json").is_file()
    assert (run.run_dir / "run_manifest.json").is_file()


def test_error_attribution_review_export_and_metrics(tmp_path):
    review_path = tmp_path / "review.csv"
    export_review_sample(
        [
            {
                "question_id": "q1",
                "question": "What is 1 + 1?",
                "gold_answer": "2",
                "test_taker_response": "3",
                "primary_error_tag": "calculation_error",
                "is_correct": False,
            }
        ],
        review_path,
        sample_size=1,
    )
    text = review_path.read_text(encoding="utf-8-sig")
    review_path.write_text(
        text.replace(
            "calculation_error,,",
            "calculation_error,calculation_error,calculation_error",
        ),
        encoding="utf-8-sig",
    )
    metrics = evaluate_review_csv(review_path)
    assert metrics["reviewed_count"] == 1
    assert metrics["per_label"]["calculation_error"]["f1"] == 1.0
    assert metrics["cohen_kappa"] == 1.0
