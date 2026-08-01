import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

import math_autobencher
from autobencher.config import (
    ConfigurationError,
    DEFAULT_TAXONOMY,
    load_project_config,
)
from autobencher.generation_guidance import (
    FORBIDDEN_GUIDANCE_FIELDS,
    build_generation_guidance_payload,
    load_generation_guidance_context,
    validate_generation_guidance,
    write_generation_guidance,
)


ROOT = Path(__file__).resolve().parents[1]

DIVERSE_STEMS = (
    "A market inventory changes twice; determine the remaining symbolic quantity",
    "Given a compact equation table, infer the one value satisfying every row",
    "A geometric construction supplies two independent measurements; "
    "compute the target",
    "Reverse a stated calculation to recover the missing initial parameter",
    "Compare two feasible cases and select the one meeting an exact boundary condition",
    "Translate a short verbal constraint into an expression before evaluating it",
    "Use a conserved total and one ratio to determine an unknown component",
    "Derive an intermediate rate, then apply it to a second interval",
    "Simplify a parameterized relation before substituting the supplied value",
    "Combine a discrete counting condition with a final arithmetic transformation",
    "Interpret a structured list of constraints and eliminate impossible candidates",
    "Use an equivalent representation to verify a result by a second computation",
)


def _source_payload(*, near_duplicates: bool = False) -> dict:
    questions = []
    for category, subcategories in DEFAULT_TAXONOMY.items():
        for subcategory in subcategories:
            for index in range(12):
                if near_duplicates:
                    question = f"Compute x + {index} for the specified value of x."
                else:
                    question = (
                        f"{DIVERSE_STEMS[index]}. Domain: {category}; "
                        f"focus: {subcategory}; marker {chr(97 + index)}."
                    )
                questions.append(
                    {
                        "question_id": f"source-{len(questions):04d}",
                        "category": category,
                        "sub_category": subcategory,
                        "difficulty": index % 10 + 1,
                        "difficulty_band": (
                            "basic"
                            if index % 3 == 0
                            else "intermediate"
                            if index % 3 == 1
                            else "advanced"
                        ),
                        "question": question,
                        "answer_type": (
                            "integer" if index % 2 else "rational"
                        ),
                        "reasoning_structure": f"structure-{index % 4}",
                        "source_dataset": f"source-{index % 3}",
                        "source_split": "test",
                        "source_revision": f"{index % 10}" * 40,
                        "source_record_id": f"record-{len(questions):04d}",
                        "source_record_sha256": f"{index % 10}" * 64,
                        "canonical_answer": str(index),
                    }
                )
    return {"questions": questions}


def _write_source(path: Path, *, near_duplicates: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_source_payload(near_duplicates=near_duplicates)),
        encoding="utf-8",
    )
    manifest_path = path.with_suffix(path.suffix + ".manifest.json")
    manifest_path.write_text(
        json.dumps(
            {
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "question_count": 324,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_builds_270_balanced_nonverbatim_guidance_records(tmp_path):
    source = _write_source(tmp_path / "candidates.json")

    first, manifest = build_generation_guidance_payload(source)
    second, _ = build_generation_guidance_payload(source)

    assert first == second
    records = validate_generation_guidance(first)
    assert len(records) == 270
    counts = Counter(
        (item["category"], item["sub_category"]) for item in records
    )
    assert len(counts) == 27
    assert set(counts.values()) == {10}
    assert first["source_text_included"] is False
    assert first["source_answers_included"] is False
    assert first["permissions"] == {
        "generator_prompt": True,
        "training": False,
        "evaluation": False,
        "hard_pool": False,
    }
    serialized = json.dumps(first, ensure_ascii=False)
    assert DIVERSE_STEMS[0] not in serialized
    assert not any(field in records[0] for field in FORBIDDEN_GUIDANCE_FIELDS)
    assert manifest["question_count"] == 270
    assert manifest["records_per_subcategory"] == 10


def test_near_duplicate_source_items_fail_closed(tmp_path):
    source = _write_source(
        tmp_path / "near_duplicates.json",
        near_duplicates=True,
    )

    with pytest.raises(ValueError, match="diversity/coverage is incomplete"):
        build_generation_guidance_payload(source)


def test_write_and_load_context_verify_manifest_and_select_three(tmp_path):
    root = tmp_path / "data"
    source = _write_source(root / "benchmarks" / "candidates.json")
    output = root / "guidance" / "guidance.json"

    manifest = write_generation_guidance(
        source,
        output,
        allowed_data_root=root,
    )
    config = {
        "generation_guidance": {
            "enabled": True,
            "dataset_path": output.as_posix(),
            "expected_record_count": 270,
            "records_per_subcategory": 10,
            "max_prompt_records": 3,
            "require_manifest": True,
        }
    }
    context, identifiers = load_generation_guidance_context(
        config,
        category="Arithmetic",
        subcategory="Integer Operations",
        target_difficulty=5,
        selection_key="coverage_deficit:5",
    )

    assert manifest["sha256"]
    assert len(identifiers) == 3
    assert all(identifier.startswith("guide-") for identifier in identifiers)
    assert "source question and answer are intentionally withheld" in context
    assert DIVERSE_STEMS[0] not in context

    output.write_text(output.read_text(encoding="utf-8") + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        load_generation_guidance_context(
            config,
            category="Arithmetic",
            subcategory="Integer Operations",
            target_difficulty=5,
            selection_key="coverage_deficit:5",
        )


def test_writer_rejects_repository_or_system_disk_outputs(tmp_path):
    root = tmp_path / "data"
    root.mkdir()
    source = _write_source(tmp_path / "outside" / "candidates.json")
    with pytest.raises(ValueError, match="under allowed_data_root"):
        write_generation_guidance(
            source,
            root / "guidance.json",
            allowed_data_root=root,
        )


def test_formal_suites_enable_guidance_but_smoke_does_not():
    suites = ROOT / "configs" / "study_suites"
    for name in ("main.yaml", "fair_budget.yaml", "ablation_round2.yaml"):
        suite = yaml.safe_load(
            (suites / name).read_text(encoding="utf-8")
        )["study_suite"]
        assert "generation_guidance.enabled=true" in suite["common_overrides"]
    smoke = yaml.safe_load(
        (suites / "smoke.yaml").read_text(encoding="utf-8")
    )["study_suite"]
    assert "generation_guidance.enabled=true" not in smoke.get(
        "common_overrides",
        [],
    )


def test_guidance_config_contract_is_strict():
    config, _ = load_project_config(
        ROOT / "configs" / "math_flywheel.yaml",
        validate_paths=False,
    )
    assert config["generation_guidance"]["expected_record_count"] == 270
    assert config["generation_guidance"]["records_per_subcategory"] == 10
    assert config["generation_guidance"]["hide_source_text"] is True

    with pytest.raises(ConfigurationError, match="must equal 27"):
        load_project_config(
            ROOT / "configs" / "math_flywheel.yaml",
            temporary_overrides=[
                "generation_guidance.expected_record_count=269"
            ],
            validate_paths=False,
        )


def test_question_generation_injects_abstract_guidance_and_records_ids(
    monkeypatch,
    tmp_path,
):
    captured = {}

    def fake_generation(**kwargs):
        captured["prompt"] = kwargs["prompt"][0]
        response = json.dumps(
            [
                {
                    "question_id": "q_1",
                    "question": "Compute 12 + 17.",
                    "answer_type": "integer",
                    "canonical_answer": "29",
                    "display_answer": "29",
                }
            ]
        )
        return SimpleNamespace(
            completions=[SimpleNamespace(text=response)]
        )

    monkeypatch.setattr(math_autobencher, "gen_from_prompt", fake_generation)
    monkeypatch.setattr(
        math_autobencher,
        "load_generation_guidance_context",
        lambda *args, **kwargs: (
            "Static abstract guide with no source question.",
            ["guide-001", "guide-002"],
        ),
    )
    monkeypatch.setattr(
        math_autobencher,
        "_validate_generated_gold_answers",
        lambda questions, *args, **kwargs: questions,
    )
    monkeypatch.setattr(
        math_autobencher,
        "_request_control_kwargs",
        lambda *args, **kwargs: {},
    )

    result = math_autobencher._generate_question_from_description(
        {
            "category": "Arithmetic",
            "sub_category": "Integer Operations",
            "difficulty": 4,
            "generation_source": "coverage_deficit",
            "generation_strategy": "quota_repair",
        },
        None,
        None,
        None,
        outfile_prefix=str(tmp_path / "generation"),
        question_count=1,
        research_config={"generation_guidance": {"enabled": True}},
    )

    assert "Static abstract guide" in captured["prompt"]
    assert "source question or answer" in " ".join(
        captured["prompt"].split()
    )
    assert result[0][0]["generation_guidance_ids"] == [
        "guide-001",
        "guide-002",
    ]
