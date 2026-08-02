import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import math_autobencher
from autobencher.attribution_eval import (
    evaluate_review_csv,
    export_blinded_review_packets,
    export_review_sample,
    merge_blinded_reviews,
)
from autobencher.config import load_resolved_config
from autobencher.dataset import (
    build_training_dataset,
    exact_signature,
    tfidf_cosine,
    token_jaccard,
    write_alpaca_jsonl,
)
from autobencher.experiment import (
    ProgressManager,
    ResearchRun,
    allocate_test_run_dir,
    atomic_json,
    build_artifact_inventory,
    run_dir_for_id,
)
from tool_util import (
    HardSamplePool,
    canonicalize_math_record,
    dump_standard_json,
    generate_math_inference,
    manage_hard_pool,
    update_hard_pool_lifecycle,
)


def test_hard_pool_lifecycle_retests_mastered_and_retires_stale(tmp_path):
    pool_path = tmp_path / "hard_pool.json"
    dump_standard_json(
        [
            {
                "unique_key": "hard-1",
                "question": "Compute 7 + 8.",
                "gold_answer": "15",
                "category": "Arithmetic",
                "sub_category": "Integer Operations",
                "sample_grade": "train_eligible",
                "lifecycle_state": "active",
                "last_seen_cycle": 1,
                "occurrences": 2,
            }
        ],
        pool_path,
    )
    for cycle, model in ((2, "model-v2"), (3, "model-v3")):
        update_hard_pool_lifecycle(
            pool_path,
            [{"unique_key": "hard-1", "is_correct": True}],
            model_version=model,
            current_cycle=cycle,
            mastered_correct_streak=2,
            stale_after_cycles=2,
            retire_after_cycles=4,
        )
    mastered = json.loads(pool_path.read_text(encoding="utf-8"))[0]
    assert mastered["lifecycle_state"] == "mastered"
    assert mastered["consecutive_correct"] == 2
    assert HardSamplePool(pool_path).get_variant_context(
        "Arithmetic", "Integer Operations"
    ) == ""
    result = update_hard_pool_lifecycle(
        pool_path,
        [],
        model_version="model-v5",
        current_cycle=5,
        retire_after_cycles=4,
    )
    assert result["state_counts"] == {"retired": 1}


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "math_flywheel_smoke_test.yaml"


@pytest.fixture
def config():
    return load_resolved_config(CONFIG_PATH)[0]


def test_test_directory_allocator_uses_maximum_existing_number(tmp_path):
    (tmp_path / "test_2").mkdir()
    (tmp_path / "test_9").mkdir()
    (tmp_path / "unrelated").mkdir()
    allocated = allocate_test_run_dir(tmp_path)
    assert allocated.name == "test_10"


def record(question, answer, *, correct=False, accuracy=0.2, **extra):
    return {
        "question": question,
        "gold_answer": answer,
        "canonical_answer": answer,
        "category": "Arithmetic",
        "sub_category": "Integer Operations",
        "answer_type": "integer",
        "difficulty": 3,
        "target_difficulty": 3,
        "observed_difficulty": 3,
        "difficulty_profile": {
            "rubric_version": "observable_math_v1",
            "score": 3,
        },
        "is_correct": correct,
        "sub_category_accuracy": accuracy,
        "evaluator_confidence": 0.95,
        "question_parse_success": True,
        "answer_validation_success": True,
        "gold_reasoning_summary": [
            f"Evaluate the stated arithmetic expression to obtain {answer}.",
            f"Check the original operation independently; it also gives {answer}.",
        ],
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
    config["training_mix"]["strict_correct_incorrect_ratio"] = False
    records = [
        record("What is 2 + 2?", "4"),
        record("What is 2 + 2?", "4"),
        record("What is 3 + 3?", "6", tool_violation=True),
    ]
    selected, manifest, rejected = build_training_dataset(records, config)
    assert len(selected) == 1
    assert manifest["rejection_reasons"]["exact_duplicate"] == 1
    assert manifest["rejection_reasons"]["tool_violation"] == 1
    assert manifest["selected_difficulty_counts"] == {"3": 1}
    assert manifest["difficulty_rubric_versions"] == [
        "observable_math_v1"
    ]
    assert len(rejected) == 2


def test_dataset_rejects_malformed_optional_numeric_fields(config):
    config["training_mix"]["strict_correct_incorrect_ratio"] = False
    dirty = record(
        "What is 8 + 1?",
        "9",
        evaluator_confidence=None,
        attribution_confidence=None,
        sub_category_accuracy=None,
    )

    selected, manifest, rejected = build_training_dataset([dirty], config)

    assert selected == []
    assert len(rejected) == 1
    assert manifest["rejection_reasons"]["low_evaluator_confidence"] == 1


def test_training_safety_rejects_blind_and_equivalence_conflicts(config):
    config["training_mix"]["strict_correct_incorrect_ratio"] = False
    blind = record(
        "Compute 41 plus 1.",
        "42",
        evaluation_role="blind_final",
        verification_tier="deterministic",
    )
    disagreement = record(
        "Compute 42 plus 1.",
        "43",
        equivalence_backend_disagreement=True,
        verification_tier="strong_heuristic",
    )
    selected, manifest, rejected = build_training_dataset(
        [blind, disagreement], config
    )
    assert selected == []
    safety = manifest["training_safety"]
    assert safety["blind_or_official_input_count"] == 1
    assert safety["blind_or_official_selected_count"] == 0
    assert safety["evaluator_disagreement_input_count"] == 1
    assert safety["evaluator_disagreement_selected_count"] == 0
    assert safety["source_attribution_tier_counts"] == {
        "deterministic": 1,
        "strong_heuristic": 1,
    }
    reasons = {reason for item in rejected for reason in item["reasons"]}
    assert "prohibited_blind_final_training_source" in reasons
    assert "equivalence_requires_review" in reasons


def test_training_sample_requires_verified_answer_and_solution_steps(config):
    config["training_mix"]["strict_correct_incorrect_ratio"] = False
    missing_steps = record(
        "Compute 91 minus 17.",
        "74",
        gold_reasoning_summary=None,
    )
    selected, manifest, rejected = build_training_dataset(
        [missing_steps],
        config,
    )
    assert selected == []
    assert manifest["rejection_reasons"][
        "missing_gold_reasoning_steps"
    ] == 1
    assert rejected[0]["reasons"] == ["missing_gold_reasoning_steps"]


def test_alpaca_output_contains_verified_steps_and_answer(config):
    config["training_mix"]["strict_correct_incorrect_ratio"] = False
    source = record(
        "Compute 12 times 7.",
        "84",
        gold_reasoning_summary=[
            "Multiply twelve by seven.",
            "Check that 84 divided by seven equals twelve.",
        ],
    )
    selected, _, _ = build_training_dataset([source], config)
    payload = json.loads(selected[0]["output"])
    assert payload["reasoning_summary"] == source["gold_reasoning_summary"]
    assert payload["final_answer"] == "84"
    assert payload["answer_type"] == "integer"


def test_canonicalization_preserves_sympy_training_reasoning(config):
    config["training_mix"]["strict_correct_incorrect_ratio"] = False
    source = record(
        "Compute 12 times 7.",
        "84",
        gold_reasoning_summary=[
            "Multiply twelve by seven to obtain 84.",
            "Check that 84 divided by seven equals twelve.",
        ],
    )

    canonical = canonicalize_math_record(source)
    selected, manifest, rejected = build_training_dataset(
        [canonical],
        config,
    )

    assert canonical["gold_reasoning_summary"] == source[
        "gold_reasoning_summary"
    ]
    assert len(selected) == 1
    assert manifest["selected_count"] == 1
    assert rejected == []


def test_training_rejects_plan_only_reasoning(config):
    config["training_mix"]["strict_correct_incorrect_ratio"] = False
    source = record(
        "Compute 12 times 7.",
        "84",
        gold_reasoning_summary=["Compute the answer.", "Check the answer."],
    )
    selected, manifest, rejected = build_training_dataset([source], config)
    assert selected == []
    assert manifest["rejection_reasons"][
        "non_concrete_gold_reasoning_steps"
    ] == 1
    assert rejected[0]["reasons"] == [
        "non_concrete_gold_reasoning_steps"
    ]


def test_alpaca_export_matches_verified_step_by_step_contract(config, tmp_path):
    config["training_mix"]["strict_correct_incorrect_ratio"] = False
    question = (
        "Solve the system for (m, n): "
        "4*m + n = 9, m - 3*n = -1."
    )
    steps = [
        "Rewrite the first equation as n = 9 - 4m.",
        "Substitute into the second equation: m - 3(9 - 4m) = -1.",
        "Simplify to 13m = 26, so m = 2.",
        "Use n = 9 - 4m with m = 2 to obtain n = 1.",
        "Check: 4*2 + 1 = 9 and 2 - 3*1 = -1, so both equations hold.",
    ]
    source = record(
        question,
        "(2, 1)",
        answer_type="ordered_tuple",
        gold_reasoning_summary=steps,
    )
    selected, _, rejected = build_training_dataset([source], config)
    assert rejected == []
    target = tmp_path / "train.jsonl"
    assert write_alpaca_jsonl(selected, target) == 1
    exported = json.loads(target.read_text(encoding="utf-8"))
    assert set(exported) == {"instruction", "input", "output"}
    assert exported["instruction"] == (
        "Solve the math problem and return a JSON object with "
        "reasoning_summary, final_answer, answer_type, and confidence."
    )
    assert exported["input"] == question
    assert json.loads(exported["output"]) == {
        "reasoning_summary": steps,
        "final_answer": "(2, 1)",
        "answer_type": "ordered_tuple",
        "confidence": 1.0,
    }


def test_dataset_template_cluster_limit(config):
    config["training_mix"]["strict_correct_incorrect_ratio"] = False
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


def test_artifact_inventory_hashes_text_and_binary(tmp_path):
    atomic_json({"status": "completed"}, tmp_path / "summary.json")
    (tmp_path / "model.bin").write_bytes(b"binary-checkpoint")
    inventory = build_artifact_inventory(tmp_path)
    assert inventory["layout_version"] == "research_run_v2"
    by_path = {item["path"]: item for item in inventory["files"]}
    assert by_path["summary.json"]["kind"] == "research_artifact"
    assert by_path["model.bin"]["kind"] == "binary_payload"
    assert all(len(item["sha256"]) == 64 for item in by_path.values())


def test_artifact_inventory_rejects_uncommitted_temporary_file(tmp_path):
    (tmp_path / "partial.json.tmp").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="temporary artifact"):
        build_artifact_inventory(tmp_path)


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


def test_hard_pool_guidance_suppresses_unverified_error_labels(tmp_path):
    pool_path = tmp_path / "hard_pool.json"
    dump_standard_json(
        [
            {
                **record("What is 2 + 3?", "5"),
                "sample_grade": "train_eligible",
                "difficulty": 2,
                "error_tags": ["sign_error"],
                "verification_tier": "abstained",
                "attribution_confidence": 0.2,
                "evidence": [{"check_name": "attribution_abstention"}],
            },
            {
                **record("What is 7 + 4?", "11"),
                "sample_grade": "train_eligible",
                "difficulty": 3,
                "error_tags": ["arithmetic_computation_error"],
                "verification_tier": "deterministic",
                "attribution_confidence": 0.98,
                "evidence": [
                    {"check_name": "reasoning_arithmetic_equality"}
                ],
            },
        ],
        pool_path,
    )
    context = HardSamplePool(pool_path).get_variant_context(
        "Arithmetic",
        "Integer Operations",
        confidence_threshold=0.70,
    )
    assert "observed_failure: arithmetic_computation_error" in context
    assert "verified_evidence_checks: reasoning_arithmetic_equality" in context
    assert "observed_failure: sign_error" not in context
    assert "observed_failure: unknown_error" in context
    assert "never reveal the reference answer" in context


def test_old_irrelevant_cache_is_reparsed_without_model_call(
    tmp_path,
    config,
    monkeypatch,
):
    question = {
        "id": 1,
        "question_id": "q1",
        "category": "Arithmetic",
        "sub_category": "Integer Operations",
        "question": "What is 2 + 2?",
        "answer_type": "integer",
        "canonical_answer": "4",
        "gold_answer": "4",
        "difficulty": 1,
    }
    cached = canonicalize_math_record(question, 0)
    cached.update(
        {
            "raw_response": (
                "Explanation.\n"
                '{"reasoning_summary":["Add."],"final_answer":"4",'
                '"answer_type":"integer","confidence":1.0}'
                "Human: unrelated"
            ),
            "parse_status": "irrelevant_output",
            "test_taker_response": "IRRELEVANT_OUTPUT",
        }
    )
    inference_path = tmp_path / "inference.json"
    dump_standard_json([cached], inference_path)

    def fail_if_called(**kwargs):
        raise AssertionError(f"model should not be called: {kwargs}")

    monkeypatch.setattr("tool_util.gen_from_prompt", fail_if_called)
    records = generate_math_inference(
        [question],
        ("unused", None, object()),
        inference_path,
        research_config=config,
    )
    assert records[0]["parse_status"] == "success"
    assert records[0]["test_taker_response"] == "4"
    assert records[0]["parser_version"] == "structured_v2"


def test_research_run_writes_reproducibility_snapshot(tmp_path, config):
    config = {**config, "paths": {**config["paths"], "output_root": str(tmp_path)}}
    provenance = {"config_hash": "abc123", "sources": {}, "schema_version": "1.0"}
    run = ResearchRun(config, provenance, "mock-run", ROOT)
    run.initialize({"config": str(CONFIG_PATH)})
    assert (run.run_dir / "resolved_config.json").is_file()
    assert (run.run_dir / "resolved_config.yaml").is_file()
    assert (run.run_dir / "config_sources.json").is_file()
    assert (run.run_dir / "config_validation.json").is_file()
    assert (run.run_dir / "config_hash.txt").is_file()
    assert run.run_dir == run_dir_for_id(tmp_path, "mock-run")
    assert (run.run_dir / "cycle").is_dir()
    assert (run.run_dir / "environment.json").is_file()
    assert (run.run_dir / "budget_ledger.json").is_file()
    manifest_path = run.run_dir / "run_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["study"]["config_snapshot"] == dict(config["study"])
    assert manifest["policy_name"] == "full"
    assert manifest["policy_version"] == manifest["study"]["policy_version"]
    assert manifest["variant"] == "full"
    assert manifest["component_state"] == manifest["study"]["component_state"]
    assert manifest["seed"] == config["experiment"]["seed"]
    assert (
        manifest["question_budget"]
        == config["experiment"]["questions_per_iteration"]
    )
    assert (
        manifest["question_budget_per_iteration"]
        == config["experiment"]["questions_per_iteration"]
    )
    assert (
        manifest["total_question_budget"]
        == config["experiment"]["questions_per_iteration"]
        * config["experiment"]["num_iterations"]
        * config["experiment"]["max_cycles"]
    )
    assert manifest["prompt_version"] == "content-addressed-v1"
    assert (
        manifest["prompt_hash"]
        == manifest["prompt_bundle"]["combined_sha256"]
    )
    assert set(manifest["prompt_bundle"]) == {
        "generator",
        "test_taker",
        "semantic_judge",
        "evaluator_solver",
        "independent_solver",
        "postcheck",
        "tora_strategy",
        "combined_sha256",
    }
    assert manifest["budget_protocol"] == "question_matched"
    with pytest.raises(ValueError, match="safe artifact name"):
        run.export_iteration(1, 1, {"../escape": []})
    with pytest.raises(ValueError, match="must be positive"):
        run.iteration_dir(0, 1)


def test_absolute_legacy_prefix_stays_inside_bound_run_dir(tmp_path):
    run_dir = tmp_path / "bound"
    outside = tmp_path / "other" / "legacy" / "output"
    prefix = math_autobencher._bound_outfile_prefix(run_dir, outside)
    assert Path(prefix).parent == run_dir
    assert Path(prefix).name == "output."


def test_research_run_resume_preserves_ledger_and_run_state(tmp_path, config):
    config = {**config, "paths": {**config["paths"], "output_root": str(tmp_path)}}
    provenance = {
        "config_hash": "resume-hash",
        "sources": {},
        "schema_version": "1.0",
    }
    first = ResearchRun(config, provenance, "resume-run", ROOT)
    first.initialize({"attempt": 1})
    first.budget_ledger.record_generation_batch(
        requested=4,
        output=3,
        accepted=2,
        failed=1,
    )
    marker = first.run_dir / "cycle" / "cycle_1" / "hard_pool.json"
    marker.parent.mkdir(parents=True)
    marker.write_text('{"preserved":true}\n', encoding="utf-8")

    resumed = ResearchRun(
        config,
        provenance,
        "resume-run",
        ROOT,
        resume=True,
    )
    resumed.initialize({"attempt": 2})
    assert resumed.run_dir == first.run_dir
    assert marker.is_file()
    assert resumed.budget_ledger.data["generation"][
        "requested_question_count"
    ] == 4
    assert resumed.budget_ledger.data["resume_count"] == 1
    manifest = json.loads(
        (resumed.run_dir / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["resume_count"] == 1


def test_resume_identity_failure_does_not_reinitialize_ledger(tmp_path, config):
    config = {**config, "paths": {**config["paths"], "output_root": str(tmp_path)}}
    provenance = {
        "config_hash": "identity-hash",
        "sources": {},
        "schema_version": "1.0",
    }
    first = ResearchRun(config, provenance, "identity-run", ROOT)
    first.initialize({"attempt": 1})
    first.budget_ledger.record_generation_batch(
        requested=2, output=2, accepted=1, failed=1
    )
    ledger_path = first.run_dir / "budget_ledger.json"
    ledger_before = ledger_path.read_bytes()
    manifest_path = first.run_dir / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["prompt_hash"] = "tampered"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RuntimeError, match="before opening mutable state"):
        ResearchRun(
            config,
            provenance,
            "identity-run",
            ROOT,
            resume=True,
        )
    assert ledger_path.read_bytes() == ledger_before


def test_resume_rejects_changed_generation_guidance_before_mutating_ledger(
    tmp_path,
    config,
):
    guidance_path = tmp_path / "guidance.json"
    guidance_path.write_text('{"version":1}\n', encoding="utf-8")
    config = {
        **config,
        "paths": {**config["paths"], "output_root": str(tmp_path)},
        "generation_guidance": {
            **config["generation_guidance"],
            "enabled": True,
            "dataset_path": str(guidance_path),
        },
    }
    provenance = {
        "config_hash": "guidance-identity-hash",
        "sources": {},
        "schema_version": "1.0",
    }
    first = ResearchRun(config, provenance, "guidance-identity-run", ROOT)
    first.initialize({"attempt": 1})
    first.budget_ledger.record_generation_batch(
        requested=2,
        output=2,
        accepted=1,
        failed=1,
    )
    ledger_path = first.run_dir / "budget_ledger.json"
    ledger_before = ledger_path.read_bytes()
    guidance_path.write_text('{"version":2}\n', encoding="utf-8")

    with pytest.raises(RuntimeError, match="before opening mutable state"):
        ResearchRun(
            config,
            provenance,
            "guidance-identity-run",
            ROOT,
            resume=True,
        )
    assert ledger_path.read_bytes() == ledger_before


def test_zero_sample_cycle_finalizes_without_undefined_iteration_state(
    tmp_path,
    config,
    monkeypatch,
    capsys,
):
    config["experiment"]["mode"] = "data_flywheel"
    config["fixed_test"]["enabled"] = False
    config["finetune"]["enabled"] = False
    cycle_root = tmp_path / "cycle"
    cycle_root.mkdir()
    finalized = {}
    research_run = SimpleNamespace(
        config=config,
        config_hash="test-config",
        cycle_root=cycle_root,
        run_dir=tmp_path,
        project_root=ROOT,
        run_id="zero-sample-test",
        metadata=lambda: {
            "run_id": "zero-sample-test",
            "config_hash": "test-config",
        },
        logger=SimpleNamespace(event=lambda *_args, **_kwargs: None),
        save_cycle_artifact=lambda *_args, **_kwargs: None,
        finalize=lambda status, summary: finalized.update(
            {"status": status, "summary": summary}
        ),
    )
    args = SimpleNamespace(
        outfile_prefix1=str(tmp_path / "math."),
        mode="data_flywheel",
        agent_modelname="agent",
        test_taker_modelname="test-taker",
        num_iters=1,
        max_cycle=1,
        export_interval=1,
        finetune_gpu="0",
        finetune_epoch=1,
        finetune_batch=1,
        lora_rank=4,
        use_helm="no",
        disk_warning_threshold=0,
        research_run=research_run,
    )

    def completed_iteration(*_args, **_kwargs):
        iteration_dir = cycle_root / "cycle_1" / "iter_1"
        iteration_dir.mkdir(parents=True)
        dump_standard_json(
            [],
            iteration_dir / "math.test_taker_inference.json",
        )
        return {"global_accuracy": 0.0, "generation_result": {}}

    empty_manifest = {
        "strict_ratio_satisfied": True,
        "minimum_sample_requirement_met": False,
        "rejection_reasons": {"missing_gold_reasoning_steps": 1},
        "template_cluster_count": 0,
        "selected_mix_counts": {},
    }
    monkeypatch.setattr(
        math_autobencher,
        "_load_test_taker_info",
        lambda *_args, **_kwargs: ("model", None, object()),
    )
    monkeypatch.setattr(
        math_autobencher,
        "_release_model_info",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        math_autobencher,
        "_run_math_iteration",
        completed_iteration,
    )
    monkeypatch.setattr(
        math_autobencher,
        "_cleanup_iteration_cache",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        math_autobencher,
        "check_disk_space",
        lambda *_args, **_kwargs: {"ok": True, "free_gb": 1.0},
    )
    monkeypatch.setattr(
        math_autobencher,
        "build_training_dataset",
        lambda *_args, **_kwargs: ([], empty_manifest, [{}]),
    )

    result = math_autobencher._run_autobencher(
        args,
        agent_info=("agent", None, object()),
        evaluator_info=("evaluator", None, object()),
    )

    assert result == 0
    assert finalized["status"] == "completed"
    assert finalized["summary"]["iteration_count"] == 1
    output = capsys.readouterr().out
    assert "[MathFlywheel] run_completed" in output
    assert '"training_sample_count": 0' in output


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


def test_blind_attribution_packets_require_consensus_or_adjudication(tmp_path):
    review_dir = tmp_path / "blind_review"
    manifest = export_blinded_review_packets(
        [
            {
                "question_id": "q1",
                "question": "What is 1 + 1?",
                "gold_answer": "2",
                "test_taker_response": "3",
                "primary_error_tag": "calculation_error",
                "attribution_confidence": 0.95,
                "is_correct": False,
            }
        ],
        review_dir,
        sample_size=400,
    )
    assert manifest["sample_size"] == 1
    packet = (review_dir / "annotator_1.csv").read_text(
        encoding="utf-8-sig"
    )
    assert "primary_error_tag" not in packet
    for number in (1, 2):
        path = review_dir / f"annotator_{number}.csv"
        text = path.read_text(encoding="utf-8-sig")
        path.write_text(
            text.replace(",,\n", ",calculation_error,\n"),
            encoding="utf-8-sig",
        )
    merged_path = review_dir / "adjudication.csv"
    assert not (review_dir / "system_predictions.sealed.csv").exists()
    assert "verification_tier" not in packet
    assert "evidence_json" not in packet
    public_manifest = json.loads(
        (review_dir / "review_manifest.json").read_text(encoding="utf-8")
    )
    assert "path" not in public_manifest["files"]["system_predictions"]
    merged = merge_blinded_reviews(
        manifest["operator_sealed_system_path"],
        review_dir / "annotator_1.csv",
        review_dir / "annotator_2.csv",
        merged_path,
    )
    assert merged["requires_adjudication"] == 0
    metrics = evaluate_review_csv(merged_path)
    assert metrics["reviewed_count"] == 1
    assert metrics["attribution_accuracy"] == 1.0
    assert metrics["acceptance"]["reviewed_count_pass"] is False
