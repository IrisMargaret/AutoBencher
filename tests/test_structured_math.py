import json
from pathlib import Path

import pytest

from autobencher.config import load_resolved_config
from autobencher.structured import (
    answers_equivalent,
    attribute_error,
    clean_answer_candidate,
    fuse_equivalence_with_semantic_judge,
    normalize_answer_type,
    normalize_generated_gold_contract,
    parse_test_taker_output,
    test_taker_prompt as strict_test_taker_prompt,
    validate_generated_question,
)


CONFIG_PATH = (
    Path(__file__).resolve().parents[1]
    / "configs"
    / "math_flywheel_smoke_test.yaml"
)


@pytest.fixture
def config():
    return load_resolved_config(CONFIG_PATH)[0]


def response(answer="8", answer_type="integer"):
    return json.dumps(
        {
            "reasoning_summary": ["Compute the requested value."],
            "final_answer": answer,
            "answer_type": answer_type,
            "confidence": 0.9,
        }
    )


def test_strict_prompt_explicitly_denies_tools(config):
    prompt = strict_test_taker_prompt(
        {"question": "What is 5 + 3?", "answer_type": "integer"},
        config,
    )
    assert "You have no tools" in prompt
    assert "Return exactly one JSON object" in prompt
    assert "return the exact symbolic form" in prompt
    assert "Never use a coarse approximation" in prompt


def test_valid_structured_response_parses(config):
    parsed = parse_test_taker_output(response(), None, "integer", config)
    assert parsed["parse_status"] == "success"
    assert parsed["parsed_response"]["final_answer"] == "8"


def test_fraction_alias_normalizes_to_rational(config):
    assert normalize_answer_type("fraction") == "rational"
    parsed = parse_test_taker_output(
        response("1/2", "fraction"),
        None,
        "rational",
        config,
    )
    assert parsed["parse_status"] == "success"
    assert parsed["parsed_response"]["answer_type"] == "rational"


def test_mixed_number_alias_normalizes_to_rational(config):
    assert normalize_answer_type("mixed_number", "1 1/2") == "rational"
    parsed = parse_test_taker_output(
        response("1 1/2", "mixed_number"),
        None,
        "rational",
        config,
    )
    assert parsed["parse_status"] == "success"
    assert parsed["parsed_response"]["answer_type"] == "rational"


@pytest.mark.parametrize(
    ("raw_type", "canonical_answer", "expected"),
    [
        ("custom_scalar", "3.25", "decimal"),
        ("exact_solution", "x = 4", "equation"),
        ("answer", "7/9", "rational"),
        ("result_kind", "12 kg", "unit_value"),
        ("custom_kind", "{1, 2}", "set"),
        ("auto", r"\frac{7}{9}", "symbolic_expression"),
        ("unknown", "plain response", "text"),
    ],
)
def test_unknown_answer_type_uses_canonical_answer_inference(
    raw_type,
    canonical_answer,
    expected,
):
    assert normalize_answer_type(raw_type, canonical_answer) == expected


def test_markdown_json_is_repaired(config):
    parsed = parse_test_taker_output(
        f"```json\n{response()}\n```", None, "integer", config
    )
    assert parsed["parse_status"] == "success"
    assert parsed["repair_attempts"] == 1


def test_tool_call_is_rejected_before_parsing(config):
    parsed = parse_test_taker_output(
        response() + '\n{"tool_calls":[]}', None, "integer", config
    )
    assert parsed["parse_status"] == "tool_violation"
    assert parsed["tool_violation"] is True


def test_role_prefixed_output_is_irrelevant(config):
    parsed = parse_test_taker_output(
        "Assistant: " + response(), None, "integer", config
    )
    assert parsed["parse_status"] == "irrelevant_output"


def test_prompt_echo_is_rejected(config):
    question = "Calculate the exact integer sum of 12345 and 67890."
    prompt = strict_test_taker_prompt(
        {"question": question, "answer_type": "integer"},
        config,
    )
    parsed = parse_test_taker_output(
        question + "\n" + response("80235"),
        prompt,
        "integer",
        config,
    )
    assert parsed["parse_status"] == "prompt_echo"
    assert parsed["contains_prompt_echo"] is True


def test_multiple_objects_are_irrelevant(config):
    parsed = parse_test_taker_output(
        response() + response(), None, "integer", config
    )
    assert parsed["parse_status"] == "irrelevant_output"


def test_first_structured_json_is_safely_extracted_from_model_overrun(config):
    raw = (
        "I will solve the problem briefly.\n"
        "```json\n"
        + response("181.8653347947", "decimal")
        + "\n```"
        "Human: Solve an unrelated equation.\n"
        "Assistant: This suffix must be discarded."
    )
    parsed = parse_test_taker_output(raw, None, "decimal", config)
    assert parsed["parse_status"] == "success"
    assert parsed["parsed_response"]["final_answer"] == "181.8653347947"
    assert parsed["extraneous_content_discarded"] is True
    assert parsed["discarded_prefix_chars"] > 0
    assert parsed["discarded_suffix_chars"] > 0
    assert parsed["contains_irrelevant_content"] is False


def test_decimal_response_rejects_symbolic_expression(config):
    parsed = parse_test_taker_output(
        response("sqrt(2)", "decimal"),
        None,
        "decimal",
        config,
    )
    assert parsed["parse_status"] == "parse_failed"


def test_excess_reasoning_is_truncated_without_losing_final_answer(config):
    reasoning = [f"Step {index}." for index in range(1, 11)]
    raw = json.dumps(
        {
            "reasoning_summary": reasoning,
            "final_answer": -71,
            "answer_type": "integer",
            "confidence": 0.0,
        }
    ) + "Human: Solve an unrelated problem."
    parsed = parse_test_taker_output(raw, None, "integer", config)
    assert parsed["parse_status"] == "success"
    assert parsed["parsed_response"]["final_answer"] == "-71"
    assert len(parsed["parsed_response"]["reasoning_summary"]) == 8
    assert parsed["reasoning_steps_truncated"] is True
    assert parsed["original_reasoning_step_count"] == 10
    assert parsed["reasoning_steps_dropped"] == 2
    assert parsed["discarded_suffix_chars"] > 0


def test_invalid_latex_escape_is_repaired_without_rejecting_answer(config):
    raw = (
        r'{"reasoning_summary":["Use \(x+1\) and simplify."],'
        '"final_answer":"2","answer_type":"integer","confidence":0.8}'
    )
    parsed = parse_test_taker_output(raw, None, "integer", config)
    assert parsed["parse_status"] == "success"
    assert parsed["parsed_response"]["final_answer"] == "2"


def test_generated_question_rejects_corrupted_unicode():
    with pytest.raises(ValueError, match="corrupted Unicode"):
        validate_generated_question(
            {
                "question_id": "q1",
                "category": "Arithmetic",
                "subcategory": "Integer Operations",
                "difficulty": 3,
                "question": "Evaluate 2 \u8133 3.",
                "answer_type": "integer",
                "canonical_answer": "6",
                "display_answer": "6",
                "unit": None,
                "tolerance": None,
                "order_sensitive": False,
                "generation_source": "coverage_deficit",
                "reference_hard_sample_ids": [],
                "target_error_type": None,
                "generation_strategy": "quota_repair",
            }
        )


@pytest.mark.parametrize(
    ("gold", "predicted", "answer_type"),
    [
        ("1/2", "0.5", "rational"),
        ("25%", "0.25", "percentage"),
        ("{1, 2, 3}", "{3, 1, 2}", "set"),
        ("x + x", "2*x", "symbolic_expression"),
        ("2*x=4", "x=2", "equation"),
        ("x > 2", "2 < x", "inequality"),
        ("[1, 3)", "[1,3)", "interval"),
        ("[[1,2],[3,4]]", "1,2;3,4", "matrix"),
        ("5 meters", "5 m", "unit_value"),
        ("log(2)", "0.6931471805599453", "symbolic_expression"),
        ("ln(2)", "0.6931471805599453", "decimal"),
        ("sqrt(2)", "1.4142135623730951", "symbolic_expression"),
        ("log(8, 2)", "3.0", "symbolic_expression"),
        ("(1/2, sqrt(2))", "(0.5, 1.4142135623730951)", "ordered_tuple"),
    ],
)
def test_answer_equivalence_types(config, gold, predicted, answer_type):
    assert answers_equivalent(
        gold, predicted, answer_type, config
    )["equivalent"] is True


@pytest.mark.parametrize(
    ("gold", "predicted", "answer_type"),
    [
        ("[[5, 4], [4, 5]]", "[[5, 6], [6, 5]]", "matrix"),
        ("[5, 5, -5]", "(5, 4, -5)", "vector"),
        ("(4, 3)", "(3, 4)", "ordered_tuple"),
        (
            "(x^2 + 2*x + 1)*exp(x)",
            "(2x + 1)*exp(x) + (x^2 + 1)*exp(x)",
            "symbolic_expression",
        ),
    ],
)
def test_typed_equivalence_rejects_structural_false_positives(
    config, gold, predicted, answer_type
):
    result = answers_equivalent(gold, predicted, answer_type, config)
    assert result["equivalent"] is False
    assert result["backend_results"]["typed"]["authoritative"] is True


def test_symbolic_implicit_multiplication_equivalent_product(config):
    result = answers_equivalent(
        "(x^2 + 2*x + 1)*exp(x)",
        "exp(x)*(x^2 + 2x + 1)",
        "symbolic_expression",
        config,
    )
    assert result["equivalent"] is True


def test_math_verify_cannot_override_typed_rejection(config, monkeypatch):
    monkeypatch.setattr(
        "autobencher.structured._math_verify_equal",
        lambda gold, predicted: (True, True),
    )
    result = answers_equivalent(
        "[[5, 4], [4, 5]]",
        "[[5, 6], [6, 5]]",
        "matrix",
        config,
    )
    assert result["equivalent"] is False
    assert result["status"] == "equivalence_backend_disagreement"
    assert result["disagreement"] is True
    assert result["needs_review"] is True


def test_semantic_judge_cannot_override_typed_rejection(config):
    equivalence = answers_equivalent("8", "9", "integer", config)
    fused = fuse_equivalence_with_semantic_judge(
        equivalence,
        judge_is_correct=True,
        judge_valid=True,
        judge_confidence=1.0,
        confidence_threshold=0.8,
        require_semantic_judge=True,
    )
    assert fused == {
        "is_correct": False,
        "status": "judge_deterministic_disagreement",
        "judge_conflict": True,
    }


def test_semantic_judge_only_resolves_typed_parse_ambiguity(config):
    equivalence = answers_equivalent(
        "an exact prose answer", "same meaning", "text", config
    )
    equivalence["deterministic_checks"]["typed_parse_success"] = False
    equivalence["status"] = "ambiguous"
    fused = fuse_equivalence_with_semantic_judge(
        equivalence,
        judge_is_correct=True,
        judge_valid=True,
        judge_confidence=0.95,
        confidence_threshold=0.8,
        require_semantic_judge=True,
    )
    assert fused["is_correct"] is True
    assert fused["status"] == "semantic_equivalent_ambiguous"


def test_exact_irrational_gold_stays_symbolic():
    contract = normalize_generated_gold_contract(
        "Find the exact diagonal of a unit square.",
        "sqrt(2)",
        "decimal",
        0.1,
    )
    assert contract["answer_type"] == "symbolic_expression"
    assert contract["canonical_answer"] == "sqrt(2)"
    assert contract["tolerance"] is None


def test_explicit_decimal_gold_is_materialized_with_fixed_tolerance():
    contract = normalize_generated_gold_contract(
        "Give sqrt(2) as a decimal to six places.",
        "sqrt(2)",
        "symbolic",
    )
    assert contract["answer_type"] == "decimal"
    assert float(contract["canonical_answer"]) == pytest.approx(2**0.5)
    assert contract["tolerance"] == pytest.approx(1.0e-3)
    assert contract["exact_canonical_answer"] == "sqrt(2)"


def test_decimal_normalizer_evaluates_symbolic_gold_only(config):
    accepted = answers_equivalent(
        "sqrt(2)",
        "1.4143",
        "decimal",
        config,
        tolerance=1.0e-3,
    )
    rejected = answers_equivalent(
        "1.4142135623731",
        "sqrt(2)",
        "decimal",
        config,
        tolerance=1.0e-3,
    )
    assert accepted["equivalent"] is True
    assert accepted["gold_normalized"]["value"] == pytest.approx(2**0.5)
    assert rejected["equivalent"] is False
    assert rejected["predicted_normalized"]["success"] is False


@pytest.mark.parametrize(
    ("raw_answer", "cleaned_answer"),
    [
        ("x = 1", "1"),
        ("x=11/7", "11/7"),
        ("135\u00b0", "135"),
        ("  281/40  ", "281/40"),
    ],
)
def test_clean_answer_candidate(raw_answer, cleaned_answer):
    assert clean_answer_candidate(raw_answer) == cleaned_answer


def test_clean_answer_candidate_preserves_numeric_zero():
    assert clean_answer_candidate(0) == "0"


def test_numeric_equivalence_cleans_prediction_without_mutating_parse(config):
    parsed = parse_test_taker_output(
        response("x = 1", "rational"),
        None,
        "rational",
        config,
    )
    original_raw_response = parsed["raw_response"]
    original_parsed_response = dict(parsed["parsed_response"])
    result = answers_equivalent(
        "1",
        parsed["parsed_response"]["final_answer"],
        "rational",
        config,
    )
    assert result["equivalent"] is True
    assert result["deterministic_checks"]["format_valid"] is True
    attribution = attribute_error(
        {"question": "Solve for the rational value of x."},
        parsed,
        result,
        config,
    )
    assert attribution["primary_error_tag"] is None
    assert parsed["raw_response"] == original_raw_response
    assert parsed["parsed_response"] == original_parsed_response
    assert parsed["parsed_response"]["final_answer"] == "x = 1"


@pytest.mark.parametrize(
    ("gold", "predicted", "answer_type"),
    [
        ("x=11/7", "11/7", "rational"),
        ("135\u00b0", "135", "decimal"),
        ("  281/40  ", "281/40", "rational"),
    ],
)
def test_gold_and_prediction_use_same_scalar_cleanup(
    config,
    gold,
    predicted,
    answer_type,
):
    result = answers_equivalent(gold, predicted, answer_type, config)
    assert result["equivalent"] is True
    assert result["deterministic_checks"]["format_valid"] is True


def test_equation_assignment_is_not_rewritten_as_numeric(config):
    result = answers_equivalent("2*x=4", "x=2", "equation", config)
    assert result["equivalent"] is True


def test_unit_mismatch_is_not_equivalent(config):
    result = answers_equivalent("5 m", "5 cm", "unit_value", config)
    assert result["equivalent"] is False
    assert result["deterministic_checks"]["unit_consistent"] is False


def test_low_confidence_attribution_becomes_unknown(config):
    parsed = {
        "parse_status": "success",
        "parsed_response": {"reasoning_summary": ["Guess."]},
    }
    equivalent = answers_equivalent("8", "9", "integer", config)
    result = attribute_error({"question": "What is 5 + 3?"}, parsed, equivalent, config)
    assert result["primary_error_tag"] == "unknown_error"
    assert result["needs_review"] is True


def test_attribution_locates_first_invalid_arithmetic_step(config):
    parsed = {
        "parse_status": "success",
        "parsed_response": {
            "reasoning_summary": [
                "First compute 5 + 3 = 9.",
                "Therefore the answer is 9.",
            ],
            "final_answer": "9",
        },
    }
    equivalent = answers_equivalent("8", "9", "integer", config)
    result = attribute_error(
        {
            "question": "What is 5 + 3?",
            "canonical_answer": "8",
            "answer_type": "integer",
        },
        parsed,
        equivalent,
        config,
    )
    assert result["primary_error_tag"] == "arithmetic_computation_error"
    assert result["first_error_step"] == 0
    assert result["attribution_confidence"] == pytest.approx(0.98)
    assert result["evidence"][0]["check_name"] == (
        "reasoning_arithmetic_equality"
    )


def test_attribution_detects_answer_transfer_after_valid_reasoning(config):
    parsed = {
        "parse_status": "success",
        "parsed_response": {
            "reasoning_summary": ["Compute 5 + 3 = 8."],
            "final_answer": "9",
        },
    }
    equivalent = answers_equivalent("8", "9", "integer", config)
    result = attribute_error(
        {
            "question": "What is 5 + 3?",
            "canonical_answer": "8",
            "answer_type": "integer",
        },
        parsed,
        equivalent,
        config,
    )
    assert result["primary_error_tag"] == "answer_transfer_error"
    assert result["evidence"][0]["check_name"] == (
        "reasoning_to_final_answer_consistency"
    )


def test_attribution_detects_wrong_irrational_approximation(config):
    parsed = {
        "parse_status": "success",
        "parsed_response": {
            "reasoning_summary": [
                "The exact diagonal is sqrt(2).",
                "Approximate it for the requested decimal output.",
            ],
            "final_answer": "1.41",
        },
    }
    equivalence = answers_equivalent(
        "1.4142135623731",
        "1.41",
        "decimal",
        config,
        tolerance=1.0e-3,
    )
    result = attribute_error(
        {
            "question": "Give the diagonal as a decimal.",
            "canonical_answer": "1.4142135623731",
            "exact_canonical_answer": "sqrt(2)",
            "answer_type": "decimal",
            "tolerance": 1.0e-3,
        },
        parsed,
        equivalence,
        config,
    )
    assert result["primary_error_tag"] == "numeric_approximation_error"
    assert result["taxonomy_version"] == "math_error_taxonomy_v4"
    assert result["evidence"][0]["check_name"] == (
        "exact_to_numeric_approximation"
    )


def test_attribution_uses_truth_solver_constraint_evidence(config):
    parsed = {
        "parse_status": "success",
        "parsed_response": {
            "reasoning_summary": ["Substitute the proposed pair."],
            "final_answer": "(1, 1)",
        },
    }
    equivalent = answers_equivalent(
        "(2, 3)",
        "(1, 1)",
        "ordered_tuple",
        config,
    )
    result = attribute_error(
        {
            "canonical_answer": "(2, 3)",
            "answer_type": "ordered_tuple",
            "test_taker_truth_validation": {
                "substitution_passed": False,
                "equations_total": 2,
                "substitution_details": [
                    {
                        "equation": "x + y = 5",
                        "passed": False,
                        "residual": "-3",
                    }
                ],
            },
        },
        parsed,
        equivalent,
        config,
    )
    assert result["primary_error_tag"] == "constraint_violation"
    assert result["evidence"][0]["check_name"] == (
        "truth_solver_substitution"
    )


def test_attribution_identifies_matrix_element_mismatch(config):
    parsed = {
        "parse_status": "success",
        "parsed_response": {"reasoning_summary": [], "final_answer": "[[1,3]]"},
    }
    equivalence = answers_equivalent("[[1,2]]", "[[1,3]]", "matrix", config)
    result = attribute_error(
        {"canonical_answer": "[[1,2]]", "answer_type": "matrix"},
        parsed,
        equivalence,
        config,
    )
    assert result["primary_error_tag"] == "matrix_element_error"
    assert result["verification_tier"] == "deterministic"
    assert result["evidence"][0]["column_index"] == 1


def test_attribution_marks_proper_subexpression_as_strong_heuristic(config):
    parsed = {
        "parse_status": "success",
        "parsed_response": {
            "reasoning_summary": [],
            "final_answer": "x + 1",
        },
    }
    equivalence = answers_equivalent(
        "(x + 1)*(x + 2)", "x + 1", "symbolic_expression", config
    )
    result = attribute_error(
        {
            "canonical_answer": "(x + 1)*(x + 2)",
            "answer_type": "symbolic_expression",
        },
        parsed,
        equivalence,
        config,
    )
    assert result["primary_error_tag"] == "incomplete_expression_evaluation"
    assert result["verification_tier"] == "strong_heuristic"
    assert result["needs_review"] is True


def test_attribution_separates_format_only_difference(config):
    parsed = {
        "parse_status": "success",
        "parsed_response": {"reasoning_summary": [], "final_answer": "2*x"},
    }
    equivalence = answers_equivalent(
        "x + x", "2*x", "symbolic_expression", config
    )
    result = attribute_error(
        {
            "canonical_answer": "x + x",
            "answer_type": "symbolic_expression",
        },
        parsed,
        equivalence,
        config,
    )
    assert result["primary_error_tag"] is None
    assert result["secondary_error_tags"] == ["format_only_difference"]


def test_attribution_rejects_choice_outside_available_options(config):
    parsed = {
        "parse_status": "success",
        "parsed_response": {
            "reasoning_summary": ["Select an option."],
            "final_answer": "Z",
        },
    }
    equivalent = answers_equivalent("B", "Z", "multiple_choice", config)
    result = attribute_error(
        {
            "canonical_answer": "B",
            "answer_type": "multiple_choice",
            "choices": ["1", "2", "3", "4"],
        },
        parsed,
        equivalent,
        config,
    )
    assert result["primary_error_tag"] == "invalid_multiple_choice"


def test_generated_question_schema_accepts_complete_record():
    validate_generated_question(
        {
            "question_id": "q1",
            "category": "Arithmetic",
            "subcategory": "Integer Operations",
            "difficulty": 1,
            "question": "What is 5 + 3?",
            "answer_type": "integer",
            "canonical_answer": "8",
            "display_answer": "8",
            "unit": None,
            "tolerance": None,
            "order_sensitive": False,
            "generation_source": "coverage_deficit",
            "reference_hard_sample_ids": [],
            "target_error_type": None,
            "generation_strategy": "quota_repair",
        }
    )


@pytest.mark.parametrize(
    "generation_source",
    [
        "hard_pool_variant",
        "coverage_deficit",
        "retention_known",
        "random",
        "uniform",
        "error_only",
    ],
)
def test_generated_question_schema_accepts_all_policy_sources(
    generation_source,
):
    validate_generated_question(
        {
            "question_id": "q-source",
            "category": "Arithmetic",
            "subcategory": "Integer Operations",
            "difficulty": 2,
            "question": "Compute 7 + 5.",
            "answer_type": "integer",
            "canonical_answer": "12",
            "display_answer": "12",
            "unit": None,
            "tolerance": None,
            "order_sensitive": False,
            "generation_source": generation_source,
            "reference_hard_sample_ids": [],
            "target_error_type": None,
            "generation_strategy": "contract_test",
        }
    )


def test_generated_question_decimal_contract_is_strict():
    base = {
        "question_id": "q-decimal",
        "category": "Geometry & Trigonometry",
        "subcategory": "Plane Geometry",
        "difficulty": 3,
        "question": "Give the diagonal as a decimal.",
        "answer_type": "decimal",
        "canonical_answer": "1.4142135623731",
        "display_answer": "1.4142135623731",
        "unit": None,
        "tolerance": 1.0e-3,
        "order_sensitive": False,
        "generation_source": "coverage_deficit",
        "reference_hard_sample_ids": [],
        "target_error_type": None,
        "generation_strategy": "quota_repair",
    }
    validate_generated_question(base)
    with pytest.raises(ValueError, match="not a symbolic expression"):
        validate_generated_question(
            {
                **base,
                "canonical_answer": "sqrt(2)",
                "display_answer": "sqrt(2)",
            }
        )
    with pytest.raises(ValueError, match="tolerance must be 0.001"):
        validate_generated_question({**base, "tolerance": None})


def test_generated_question_symbolic_contract_has_no_tolerance():
    base = {
        "question_id": "q-symbolic",
        "category": "Geometry & Trigonometry",
        "subcategory": "Plane Geometry",
        "difficulty": 3,
        "question": "Give the exact diagonal.",
        "answer_type": "symbolic_expression",
        "canonical_answer": "sqrt(2)",
        "display_answer": "sqrt(2)",
        "unit": None,
        "tolerance": None,
        "order_sensitive": False,
        "generation_source": "coverage_deficit",
        "reference_hard_sample_ids": [],
        "target_error_type": None,
        "generation_strategy": "quota_repair",
    }
    validate_generated_question(base)
    with pytest.raises(ValueError, match="must not define"):
        validate_generated_question({**base, "tolerance": 1.0e-3})


def test_generated_question_rejects_inconsistent_difficulty_profile():
    with pytest.raises(ValueError, match="observed_difficulty"):
        validate_generated_question(
            {
                "question_id": "q1",
                "category": "Arithmetic",
                "subcategory": "Integer Operations",
                "difficulty": 2,
                "target_difficulty": 3,
                "observed_difficulty": 4,
                "difficulty_profile": {
                    "rubric_version": "observable_math_v1",
                    "score": 2,
                    "band": "foundational",
                    "confidence": 0.9,
                    "effective_score": 2,
                    "requested_score": 3,
                    "dimensions": {
                        "reasoning_steps": {},
                        "operation_count": {},
                        "constraint_count": {},
                        "symbolic_depth": {},
                        "representation_load": {},
                    },
                },
                "question": "What is 5 + 3?",
                "answer_type": "integer",
                "canonical_answer": "8",
                "display_answer": "8",
                "unit": None,
                "tolerance": None,
                "order_sensitive": False,
                "generation_source": "coverage_deficit",
                "reference_hard_sample_ids": [],
                "target_error_type": None,
                "generation_strategy": "quota_repair",
            }
        )


def test_generated_question_schema_rejects_unknown_category():
    with pytest.raises(ValueError, match="validation failed"):
        validate_generated_question(
            {
                "question_id": "q1",
                "category": "Unknown",
                "subcategory": "Integer Operations",
                "difficulty": 1,
                "question": "What is 5 + 3?",
                "answer_type": "integer",
                "canonical_answer": "8",
                "display_answer": "8",
                "unit": None,
                "tolerance": None,
                "order_sensitive": False,
                "generation_source": "coverage_deficit",
                "reference_hard_sample_ids": [],
                "target_error_type": None,
                "generation_strategy": "quota_repair",
            }
        )
