import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from autobencher.config import DEFAULT_TAXONOMY, load_resolved_config
from autobencher.dataset import (
    build_training_dataset,
    holdout_leakage_reason,
)
from autobencher.evaluator import (
    EvaluatorProtocolError,
    execute_generated_python,
    judge_answer_semantics,
    solve_with_privileged_python,
    validate_generated_python,
)
from autobencher.fixed_benchmark import load_fixed_test_set
from autobencher.similarity import (
    MinHashBackendUnavailable,
    SemanticSimilarityUnavailable,
    build_similarity_batch,
    minhash_signature,
    minhash_similarity,
)
from autobencher.structured import _math_verify_equal


ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "math_flywheel_smoke_test.yaml"


@pytest.fixture
def config():
    return load_resolved_config(CONFIG_PATH)[0]


def _training_record(index, correct):
    return {
        "question": f"Unique training exercise {index}: compute {index} + 11.",
        "gold_answer": str(index + 11),
        "canonical_answer": str(index + 11),
        "category": "Arithmetic",
        "sub_category": "Integer Operations",
        "answer_type": "integer",
        "is_correct": correct,
        "sub_category_accuracy": 0.2,
        "evaluator_confidence": 0.99,
        "question_parse_success": True,
        "answer_validation_success": True,
        "gold_reasoning_summary": [
            "Compute the requested expression and verify the result."
        ],
    }


def test_fixed_test_covers_every_subcategory(config):
    questions, metadata = load_fixed_test_set(config, ROOT)
    expected = {
        (category, subcategory)
        for category, subcategories in DEFAULT_TAXONOMY.items()
        for subcategory in subcategories
    }
    observed = {
        (record["category"], record["sub_category"])
        for record in questions
    }
    assert len(questions) == 27
    assert observed == expected
    assert metadata["covered_subcategory_count"] == 27
    assert len(metadata["sha256"]) == 64


def test_math_verify_fraction_decimal_equivalence_without_worker_timeout():
    available, equivalent = _math_verify_equal("1/2", "0.5")
    if not available:
        pytest.skip("optional math-verify dependency is unavailable")
    assert equivalent is True


def test_strict_training_mix_is_exactly_25_75(config):
    config["dataset"]["text_near_dedup"] = False
    config["dataset"]["semantic_dedup"] = False
    config["dataset"]["template_dedup"] = False
    records = [
        _training_record(index, correct=index < 4)
        for index in range(16)
    ]
    selected, manifest, _ = build_training_dataset(records, config)
    assert len(selected) == 16
    assert manifest["selected_correct_count"] == 4
    assert manifest["selected_incorrect_count"] == 12
    assert manifest["selected_correct_fraction"] == 0.25
    assert manifest["strict_ratio_satisfied"] is True


def test_holdout_exact_and_template_leakage_are_rejected(config):
    holdout = [
        {
            "question_id": "fixed-a",
            "question": "Solve for x: 3*x + 2 = 11.",
        }
    ]
    reason, _ = holdout_leakage_reason(
        {"question": "Solve for x: 3*x + 2 = 11."},
        holdout,
        config,
    )
    assert reason == "holdout_exact_match"
    reason, details = holdout_leakage_reason(
        {"question": "Solve for x: 8*x + 4 = 20."},
        holdout,
        config,
    )
    assert reason == "holdout_template_match"
    assert details["holdout_question_id"] == "fixed-a"


def test_text_dedup_minhash_detects_near_duplicate_math_questions(config):
    options = config["dataset"]
    left = minhash_signature(
        "Solve the equation x + 2 = 5.",
        num_perm=options["text_dedup_num_perm"],
        ngram_size=options["text_dedup_ngram_size"],
        seed=options["text_dedup_seed"],
    )
    right = minhash_signature(
        "Solve the equation x + 2 = 6.",
        num_perm=options["text_dedup_num_perm"],
        ngram_size=options["text_dedup_ngram_size"],
        seed=options["text_dedup_seed"],
    )
    unrelated = minhash_signature(
        "Find the area of a circle with radius ten.",
        num_perm=options["text_dedup_num_perm"],
        ngram_size=options["text_dedup_ngram_size"],
        seed=options["text_dedup_seed"],
    )
    close_score = minhash_similarity(left, right)
    assert 0.50 <= close_score < 1.0
    assert close_score > minhash_similarity(left, unrelated)


def test_required_datasketch_backend_fails_closed(config):
    config["dataset"].update(
        {
            "text_dedup_enabled": True,
            "datasketch_enabled": True,
            "datasketch_required": True,
        }
    )
    with patch(
        "autobencher.similarity._datasketch_minhash",
        side_effect=MinHashBackendUnavailable("datasketch unavailable"),
    ), pytest.raises(MinHashBackendUnavailable, match="datasketch unavailable"):
        build_similarity_batch(
            ["Solve x + 2 = 5.", "Solve x + 2 = 6."],
            config["dataset"],
        )


def test_datasketch_backend_records_lsh_candidates(config):
    class FakeSketch:
        def __init__(self, value):
            self.hashvalues = np.asarray(value, dtype=np.uint64)

    class FakeLSH:
        def __init__(self, **kwargs):
            self.items = {}

        def insert(self, key, sketch):
            self.items[key] = sketch

        def query(self, sketch):
            del sketch
            return list(self.items)

    config["dataset"].update(
        {
            "text_dedup_enabled": True,
            "datasketch_enabled": True,
            "datasketch_required": True,
            "datasketch_lsh_enabled": True,
        }
    )
    fake_module = type(
        "FakeDatasketch",
        (),
        {"MinHashLSH": FakeLSH},
    )
    signatures = (
        (tuple(range(8)), FakeSketch(range(8))),
        (tuple(range(8)), FakeSketch(range(8))),
    )
    with patch.dict("sys.modules", {"datasketch": fake_module}), patch(
        "autobencher.similarity._datasketch_minhash",
        side_effect=signatures,
    ):
        batch = build_similarity_batch(
            ["Solve x + 2 = 5.", "Solve x + 2 = 6."],
            {
                **config["dataset"],
                "text_dedup_num_perm": 8,
            },
        )
    assert batch.minhash_backend == "datasketch"
    assert batch.minhash_candidates(0) == (1,)


def test_sentence_transformers_semantic_holdout_rejection(config):
    class FakeSentenceTransformer:
        def encode(self, texts, **kwargs):
            del kwargs
            return np.ones((len(texts), 4), dtype=np.float32) / 2

    config["dataset"].update(
        {
            "sentence_transformers_enabled": True,
            "sentence_transformers_required": True,
            "text_dedup_enabled": False,
        }
    )
    with patch(
        "autobencher.similarity._load_sentence_transformer",
        return_value=FakeSentenceTransformer(),
    ):
        reason, details = holdout_leakage_reason(
            {"question": "Determine the unknown value in this relation."},
            [
                {
                    "question_id": "fixed-semantic",
                    "question": "Find the missing quantity from the equation.",
                }
            ],
            config,
        )
    assert reason == "holdout_semantic_duplicate"
    assert details["sentence_transformers_similarity"] == pytest.approx(1.0)


def test_required_sentence_transformers_backend_fails_closed(config):
    config["dataset"].update(
        {
            "sentence_transformers_enabled": True,
            "sentence_transformers_required": True,
        }
    )
    with patch(
        "autobencher.similarity._load_sentence_transformer",
        side_effect=SemanticSimilarityUnavailable("model unavailable"),
    ), pytest.raises(SemanticSimilarityUnavailable, match="model unavailable"):
        holdout_leakage_reason(
            {"question": "Compute a quantity from unrelated givens."},
            [
                {
                    "question_id": "fixed-backend",
                    "question": "Find a result for a distinct problem.",
                }
            ],
            config,
        )


def test_generated_python_is_isolated_and_executes():
    code = """
primary_answer = sp.simplify(Fraction(3, 4) + Fraction(5, 6))
independent_answer = Fraction(19, 12)
verification_passed = primary_answer == independent_answer
substitution_passed = primary_answer - Fraction(5, 6) == Fraction(3, 4)
result = {
    "canonical_answer": str(primary_answer),
    "answer_type": "rational",
    "verification_passed": verification_passed,
    "substitution_passed": substitution_passed,
    "verification_details": ["independent fraction subtraction passed"],
}
"""
    validated = validate_generated_python(code, 12000)
    result = execute_generated_python(
        validated,
        timeout_seconds=10,
        max_output_chars=16000,
    )
    assert result["canonical_answer"] == "19/12"
    assert result["verification_passed"] is True
    with pytest.raises(EvaluatorProtocolError, match="imports"):
        validate_generated_python("import os\nresult = {}", 12000)
    with pytest.raises(EvaluatorProtocolError, match="computed variables"):
        validate_generated_python(
            "result = {"
            "'canonical_answer': '2', 'answer_type': 'integer', "
            "'verification_passed': True, 'substitution_passed': True, "
            "'verification_details': []}",
            12000,
        )


def test_privileged_solver_requires_runtime_and_postcheck(config, tmp_path):
    proposal = {
        "analysis_summary": ["compute", "substitute"],
        "python_code": (
            "primary_answer = "
            "sp.solve(sp.Eq(3 * sp.Symbol('x') + 2, 11))[0]\n"
            "independent_answer = Fraction(11 - 2, 3)\n"
            "verification_passed = primary_answer == independent_answer\n"
            "substitution_passed = 3*primary_answer + 2 == 11\n"
            "result = {"
            "\"canonical_answer\": str(primary_answer), "
            "\"answer_type\": \"integer\", "
            "\"verification_passed\": verification_passed, "
            "\"substitution_passed\": substitution_passed, "
            "\"verification_details\": [\"3*3+2=11\"]}"
        ),
    }
    postcheck = {
        "accepted": True,
        "verified_answer": "3",
        "answer_type": "integer",
        "substitution_passed": True,
        "difficulty_acceptable": True,
        "estimated_difficulty": 2,
        "reason": "Substitution gives 11.",
    }
    with patch(
        "autobencher.evaluator._model_json",
        side_effect=[proposal, proposal, postcheck],
    ):
        result = solve_with_privileged_python(
            "Solve for x: 3*x + 2 = 11.",
            ("model", None, object()),
            config,
            cache_path=tmp_path / "solver.json",
        )
    assert result["status"] == "passed"
    assert result["canonical_answer"] == "3"
    assert result["substitution_passed"] is True
    cache = json.loads(
        (tmp_path / "solver.json").read_text(encoding="utf-8")
    )
    assert next(iter(cache.values()))["python_code_sha256"]
    assert next(iter(cache.values()))["independent_python_code_sha256"]


def test_privileged_solver_rejects_unrelated_gold_text(config, tmp_path):
    bad_proposal = {
        "analysis_summary": ["compute", "check"],
        "python_code": (
            "primary_answer = 'Assistant: unrelated characters'\n"
            "independent_answer = 'Assistant:' + ' unrelated characters'\n"
            "verification_passed = primary_answer == independent_answer\n"
            "substitution_passed = len(primary_answer) == len(independent_answer)\n"
            "result = {"
            "\"canonical_answer\": primary_answer, "
            "\"answer_type\": \"text\", "
            "\"verification_passed\": verification_passed, "
            "\"substitution_passed\": substitution_passed, "
            "\"verification_details\": [\"string lengths match\"]}"
        ),
    }
    with patch(
        "autobencher.evaluator._model_json",
        return_value=bad_proposal,
    ):
        result = solve_with_privileged_python(
            "Compute 2 + 2.",
            ("model", None, object()),
            config,
            cache_path=tmp_path / "bad-solver.json",
        )
    assert result["status"] == "failed"
    assert "prompt, role, code, or Markdown" in result["failure_reason"]


def test_semantic_judge_uses_strict_model_result(config):
    model_result = {
        "semantically_equivalent": True,
        "confidence": 0.98,
        "reason": "Both values equal one half.",
        "format_only_difference": True,
    }
    with patch(
        "autobencher.evaluator._model_json",
        return_value=model_result,
    ):
        result = judge_answer_semantics(
            question="Compute 1/2.",
            gold_answer="1/2",
            predicted_answer="0.5",
            answer_type="rational",
            evaluator_info=("model", None, object()),
            config=config,
        )
    assert result["status"] == "success"
    assert result["semantically_equivalent"] is True
    assert result["prompt_version"]
