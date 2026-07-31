import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from autobencher.config import DEFAULT_TAXONOMY
from autobencher.open_source_benchmark import (
    _load_source_records,
    build_open_source_candidate_payload,
    write_open_source_candidate_set,
)


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "configs" / "open_source_evaluation_sources.yaml"


def test_open_source_catalog_is_pinned_and_never_uses_huggingface_urls():
    payload = yaml.safe_load(CATALOG.read_text(encoding="utf-8"))
    assert payload["candidate_set"]["questions_per_subcategory"] == 20
    assert {item["id"] for item in payload["sources"]} == {
        "gsm8k",
        "hendrycks_math",
        "mmlu_math",
        "deepmind_mathematics",
    }
    for source in payload["sources"]:
        assert len(source["revision"]) == 40
        assert set(source["revision"]) <= set("0123456789abcdef")
        assert source["repository"].startswith("https://github.com/")
        assert "huggingface.co" not in source["repository"]
        assert not Path(source["path"]).is_absolute()


def test_gsm8k_import_preserves_provenance_and_stays_pending(tmp_path):
    source_root = tmp_path / "sources"
    source_file = source_root / "gsm8k" / "test.jsonl"
    source_file.parent.mkdir(parents=True)
    source_file.write_text(
        json.dumps(
            {
                "question": (
                    "A cyclist travels at 18 km/h for 2 hours. "
                    "How far does the cyclist travel?"
                ),
                "answer": "18 * 2 = 36\n#### 36",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    source = {
        "id": "gsm8k",
        "adapter": "gsm8k_jsonl",
        "repository": "https://github.com/openai/grade-school-math",
        "revision": "3" * 40,
        "license": "MIT",
        "split": "test",
        "path": "gsm8k/test.jsonl",
    }

    records, files = _load_source_records(source, source_root.resolve())

    assert len(records) == 1
    assert records[0]["source_dataset"] == "gsm8k"
    assert records[0]["source_split"] == "test"
    assert records[0]["canonical_answer"] == "36"
    assert records[0]["validation"]["status"] == (
        "pending_independent_verification"
    )
    assert records[0]["taxonomy_assignment"]["status"] == (
        "requires_human_review"
    )
    assert files[0]["sha256"]


@pytest.mark.parametrize(
    ("adapter", "relative_path", "contents", "extra", "expected_type"),
    [
        (
            "hendrycks_math_json",
            "math/test/algebra/1.json",
            json.dumps(
                {
                    "problem": "Solve for x: x + 1 = 3.",
                    "solution": "Subtract one, so \\boxed{2}.",
                    "level": "Level 2",
                    "type": "Algebra",
                }
            ),
            {},
            "integer",
        ),
        (
            "mmlu_csv",
            "mmlu/test/elementary_mathematics_test.csv",
            "What is 1+1?,1,2,3,4,B\n",
            {"subjects": ["elementary_mathematics"]},
            "multiple_choice",
        ),
        (
            "deepmind_text",
            "deepmind/test/arithmetic__add_or_sub.txt",
            "What is 7 + 5?\n12\n",
            {},
            "integer",
        ),
    ],
)
def test_upstream_adapters_parse_their_native_static_formats(
    tmp_path,
    adapter,
    relative_path,
    contents,
    extra,
    expected_type,
):
    source_root = tmp_path / "sources"
    source_file = source_root / relative_path
    source_file.parent.mkdir(parents=True)
    source_file.write_text(contents, encoding="utf-8")
    source = {
        "id": adapter,
        "adapter": adapter,
        "repository": "https://github.com/example/source",
        "revision": "4" * 40,
        "license": "MIT",
        "split": "test",
        "path": str(Path(relative_path).parent).replace("\\", "/"),
        **extra,
    }

    records, _ = _load_source_records(source, source_root.resolve())

    assert len(records) == 1
    assert records[0]["answer_type"] == expected_type
    assert records[0]["source_record_sha256"]


def test_candidate_builder_is_deterministic_balanced_and_not_release_ready(
    tmp_path,
    monkeypatch,
):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "candidate_set": {
                    "name": "fixture_candidates",
                    "version": "1.0",
                    "seed": 42,
                    "questions_per_subcategory": 20,
                },
                "sources": [
                    {
                        "id": "fixture",
                        "adapter": "gsm8k_jsonl",
                        "repository": "https://github.com/example/fixture",
                        "revision": "a" * 40,
                        "license": "MIT",
                        "split": "test",
                        "path": "unused.jsonl",
                    }
                ],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    fixture_records = []
    for category, subcategories in DEFAULT_TAXONOMY.items():
        for subcategory in subcategories:
            for index in range(20):
                fixture_records.append(
                    {
                        "question_id": f"q-{len(fixture_records):04d}",
                        "category": category,
                        "sub_category": subcategory,
                        "difficulty": index % 10 + 1,
                        "difficulty_band": (
                            "basic" if index % 3 == 0
                            else "intermediate" if index % 3 == 1
                            else "advanced"
                        ),
                        "question": (
                            f"{category} {subcategory} unique_marker_{index}"
                        ),
                        "answer_type": "integer" if index % 2 else "rational",
                        "canonical_answer": str(index),
                        "display_answer": str(index),
                        "tolerance": None,
                        "reasoning_structure": f"structure-{index % 3}",
                        "template_cluster": f"template-{category}-{subcategory}-{index}",
                        "source_dataset": "fixture",
                        "source_split": "test",
                        "source_record_id": str(index),
                        "source_revision": "a" * 40,
                        "source_repository": "https://github.com/example/fixture",
                        "source_license": "MIT",
                        "source_record_sha256": "b" * 64,
                        "taxonomy_assignment": {
                            "status": "requires_human_review",
                            "rule": "fixture",
                        },
                        "validation": {
                            "status": "pending_independent_verification",
                            "sources": [],
                        },
                    }
                )

    monkeypatch.setattr(
        "autobencher.open_source_benchmark._load_source_records",
        lambda source, root: (fixture_records, []),
    )
    first, manifest = build_open_source_candidate_payload(catalog, source_root)
    second, _ = build_open_source_candidate_payload(catalog, source_root)

    assert first == second
    assert len(first["questions"]) == 540
    counts = Counter(
        (item["category"], item["sub_category"])
        for item in first["questions"]
    )
    assert len(counts) == 27
    assert set(counts.values()) == {20}
    assert first["release_status"].startswith("pending_")
    assert manifest["release_ready"] is False


def test_candidate_builder_fails_closed_on_coverage_shortage(
    tmp_path,
    monkeypatch,
):
    source_root = tmp_path / "sources"
    source_root.mkdir()
    catalog = tmp_path / "catalog.yaml"
    catalog.write_text(
        yaml.safe_dump(
            {
                "schema_version": "1.0",
                "candidate_set": {
                    "name": "incomplete",
                    "version": "1.0",
                    "seed": 1,
                    "questions_per_subcategory": 20,
                },
                "sources": [
                    {
                        "id": "fixture",
                        "adapter": "gsm8k_jsonl",
                        "repository": "https://github.com/example/fixture",
                        "revision": "a" * 40,
                        "license": "MIT",
                        "split": "test",
                        "path": "unused.jsonl",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "autobencher.open_source_benchmark._load_source_records",
        lambda source, root: ([], []),
    )

    with pytest.raises(ValueError, match="coverage is incomplete"):
        build_open_source_candidate_payload(catalog, source_root)


def test_candidate_writer_rejects_inputs_outside_data_root(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(ValueError, match="inputs must remain"):
        write_open_source_candidate_set(
            CATALOG,
            outside,
            data_root / "benchmarks" / "candidate.json",
            data_root,
        )
