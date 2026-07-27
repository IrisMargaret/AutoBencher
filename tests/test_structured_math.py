import json
from pathlib import Path

import pytest

from autobencher.config import load_resolved_config
from autobencher.structured import (
    answers_equivalent,
    attribute_error,
    normalize_answer_type,
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
        + response("105*sqrt(3)", "decimal")
        + "\n```"
        "Human: Solve an unrelated equation.\n"
        "Assistant: This suffix must be discarded."
    )
    parsed = parse_test_taker_output(raw, None, "decimal", config)
    assert parsed["parse_status"] == "success"
    assert parsed["parsed_response"]["final_answer"] == "105*sqrt(3)"
    assert parsed["extraneous_content_discarded"] is True
    assert parsed["discarded_prefix_chars"] > 0
    assert parsed["discarded_suffix_chars"] > 0
    assert parsed["contains_irrelevant_content"] is False


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
    ],
)
def test_answer_equivalence_types(config, gold, predicted, answer_type):
    assert answers_equivalent(
        gold, predicted, answer_type, config
    )["equivalent"] is True


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
