"""Structured model output, answer normalization, equivalence, and attribution."""

from __future__ import annotations

import ast
import json
import math
import re
import unicodedata
from fractions import Fraction
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator


ANSWER_TYPES = {
    "integer",
    "decimal",
    "rational",
    "percentage",
    "boolean",
    "text",
    "symbolic_expression",
    "equation",
    "inequality",
    "set",
    "interval",
    "ordered_tuple",
    "unordered_collection",
    "vector",
    "matrix",
    "unit_value",
    "multiple_choice",
}

ANSWER_TYPE_ALIASES = {
    "int": "integer",
    "whole_number": "integer",
    "float": "decimal",
    "number": "decimal",
    "numeric": "decimal",
    "fraction": "rational",
    "ratio": "rational",
    "percent": "percentage",
    "bool": "boolean",
    "string": "text",
    "free_text": "text",
    "expression": "symbolic_expression",
    "algebraic_expression": "symbolic_expression",
    "ordered_pair": "ordered_tuple",
    "tuple": "ordered_tuple",
    "list": "unordered_collection",
    "collection": "unordered_collection",
    "array": "vector",
    "quantity": "unit_value",
    "quantity_with_unit": "unit_value",
    "choice": "multiple_choice",
}

ERROR_TAGS = (
    "concept_confusion",
    "formula_memory_error",
    "calculation_error",
    "multi_step_logic_error",
    "condition_missing",
    "format_output_error",
    "tool_violation",
    "irrelevant_output",
    "prompt_echo",
    "parse_failed",
    "unknown_error",
)

TOOL_PATTERNS = (
    r"<tool_call>",
    r'"tool_calls"\s*:',
    r'"function"\s*:',
    r"\bpython\s*\(",
    r"\bsympy\.",
    r"\bcalculator\b",
    r"\bbrowse(?:r)?\b",
    r"\bsearch_engine\b",
)

ROLE_PREFIX_PATTERN = re.compile(
    r"^\s*(?:Human|User|System|Assistant)\s*:",
    flags=re.IGNORECASE,
)

SCHEMA_ROOT = Path(__file__).resolve().parents[1] / "schemas"


def normalize_answer_type(value: Any) -> str:
    """Map common model-produced aliases onto the versioned answer schema."""
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value or "text").strip().lower(),
    ).strip("_")
    return ANSWER_TYPE_ALIASES.get(normalized, normalized)


def validate_json_schema(payload: Any, schema_name: str) -> None:
    """Validate an artifact against a checked-in Draft 2020-12 schema."""
    schema_path = SCHEMA_ROOT / schema_name
    with schema_path.open("r", encoding="utf-8") as handle:
        schema = json.load(handle)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(payload),
        key=lambda error: tuple(str(part) for part in error.absolute_path),
    )
    if errors:
        details = "; ".join(
            (
                f"{'/'.join(str(part) for part in error.absolute_path) or '<root>'}: "
                f"{error.message}"
            )
            for error in errors[:5]
        )
        raise ValueError(f"{schema_name} validation failed: {details}")


def validate_generated_question(payload: Mapping[str, Any]) -> None:
    """Validate the strict generated-question contract."""
    validate_json_schema(dict(payload), "generated_question.schema.json")


def test_taker_prompt(question: Mapping[str, Any], config: Mapping[str, Any]) -> str:
    prompt_config = config["test_taker_prompt"]
    answer_type = normalize_answer_type(question.get("answer_type", "text"))
    return f"""You are the test-taker model. You have no tools.
Use only your internal mathematical reasoning. Never call Python, a calculator,
SymPy, search, files, a browser, an API, or any external tool.

Return exactly one JSON object and no other text:
{{
  "reasoning_summary": ["one concise auditable step"],
  "final_answer": "standalone final answer",
  "answer_type": "{answer_type}",
  "confidence": 0.0
}}

Constraints:
- reasoning_summary must contain {int(prompt_config['min_reasoning_steps'])} to
  {int(prompt_config['max_reasoning_steps'])} short steps.
- Each step must contain at most {int(prompt_config['max_chars_per_step'])} characters.
- Do not echo this prompt or the question.
- Do not output Markdown, role prefixes, extra questions, or tool calls.
- final_answer must be independent and match answer_type.

Question:
{question.get('question', '')}
"""


def _repair_json_text(text: str) -> str:
    repaired = text.strip().lstrip("\ufeff")
    repaired = re.sub(r"^```(?:json)?\s*", "", repaired, flags=re.IGNORECASE)
    repaired = re.sub(r"\s*```$", "", repaired)
    repaired = repaired.translate(
        str.maketrans({"“": '"', "”": '"', "‘": "'", "’": "'"})
    )
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    repaired = re.sub(
        r"([,{]\s*)([A-Za-z_][A-Za-z0-9_-]*)(\s*:)",
        r'\1"\2"\3',
        repaired,
    )
    return repaired


def _extract_single_object(text: str) -> tuple[str | None, str, bool]:
    decoder = json.JSONDecoder()
    candidates = []
    for match in re.finditer(r"\{", text):
        try:
            value, end = decoder.raw_decode(text[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((match.start(), match.start() + end))
    if len(candidates) != 1:
        return None, text, len(candidates) > 1
    start, end = candidates[0]
    surrounding = (text[:start] + text[end:]).strip()
    return text[start:end], surrounding, False


def _contains_tool_call(text: str) -> bool:
    return any(re.search(pattern, text, flags=re.IGNORECASE) for pattern in TOOL_PATTERNS)


def _prompt_echo(raw: str, prompt: str | None) -> bool:
    if not prompt:
        return False
    normalized_prompt = re.sub(r"\s+", " ", prompt).strip().lower()
    normalized_raw = re.sub(r"\s+", " ", raw).strip().lower()
    if len(normalized_prompt) >= 40 and normalized_prompt[:120] in normalized_raw:
        return True
    question_match = re.search(r"Question:\s*(.+)", prompt, flags=re.DOTALL)
    if question_match:
        question = re.sub(r"\s+", " ", question_match.group(1)).strip().lower()
        return len(question) >= 20 and question[:100] in normalized_raw
    return False


def parse_test_taker_output(
    raw_response: str,
    prompt: str | None,
    expected_answer_type: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    raw = str(raw_response or "").strip()
    result = {
        "raw_response": raw,
        "parsed_response": {},
        "parse_status": "parse_failed",
        "repair_attempts": 0,
        "contains_prompt_echo": _prompt_echo(raw, prompt),
        "contains_irrelevant_content": False,
        "tool_violation": _contains_tool_call(raw),
    }
    if result["tool_violation"]:
        result["parse_status"] = "tool_violation"
        return result
    if result["contains_prompt_echo"]:
        result["parse_status"] = "prompt_echo"
        return result
    if ROLE_PREFIX_PATTERN.search(raw):
        result["contains_irrelevant_content"] = True
        result["parse_status"] = "irrelevant_output"
        return result
    max_repairs = int(config["test_taker_prompt"]["format_repair_attempts"])
    attempts = [raw]
    if max_repairs > 0:
        attempts.append(_repair_json_text(raw))
    for attempt, candidate in enumerate(attempts[: max_repairs + 1]):
        object_text, surrounding, multiple = _extract_single_object(candidate)
        if multiple:
            result["contains_irrelevant_content"] = True
            result["parse_status"] = "irrelevant_output"
            return result
        if object_text is None:
            object_text = candidate if candidate.startswith("{") else None
        if object_text is None:
            continue
        try:
            parsed = json.loads(object_text)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(object_text)
            except (ValueError, SyntaxError):
                continue
        if not isinstance(parsed, dict):
            continue
        required = {"reasoning_summary", "final_answer", "answer_type", "confidence"}
        if not required.issubset(parsed):
            continue
        reasoning = parsed["reasoning_summary"]
        if not isinstance(reasoning, list) or not all(
            isinstance(step, str) and step.strip() for step in reasoning
        ):
            continue
        prompt_config = config["test_taker_prompt"]
        if not (
            int(prompt_config["min_reasoning_steps"])
            <= len(reasoning)
            <= int(prompt_config["max_reasoning_steps"])
        ):
            continue
        if any(
            len(step) > int(prompt_config["max_chars_per_step"])
            for step in reasoning
        ):
            continue
        parsed_answer_type = normalize_answer_type(parsed["answer_type"])
        expected_type = normalize_answer_type(expected_answer_type)
        if parsed_answer_type != expected_type:
            continue
        try:
            confidence = float(parsed["confidence"])
        except (TypeError, ValueError):
            continue
        if not 0 <= confidence <= 1:
            continue
        if not str(parsed["final_answer"]).strip():
            continue
        if surrounding:
            if attempt < min(max_repairs, len(attempts) - 1):
                continue
            result["contains_irrelevant_content"] = True
            result["parse_status"] = "irrelevant_output"
            return result
        result["parsed_response"] = {
            "reasoning_summary": [step.strip() for step in reasoning],
            "final_answer": str(parsed["final_answer"]).strip(),
            "answer_type": parsed_answer_type,
            "confidence": confidence,
        }
        result["parse_status"] = "success"
        result["repair_attempts"] = attempt
        return result
    return result


def _clean_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", str(value or ""))
    text = text.strip().strip("`")
    text = re.sub(r"^(?:answer|final answer)\s*[:=]\s*", "", text, flags=re.I)
    text = re.sub(r"\s+", " ", text)
    return text.strip(" \t\r\n.;")


def _number(value: Any, percentage: bool = False) -> float | None:
    text = _clean_text(value).replace(",", "")
    has_percent_sign = text.endswith("%")
    if has_percent_sign:
        text = text[:-1].strip()
    try:
        number = float(Fraction(text))
    except (ValueError, ZeroDivisionError):
        try:
            number = float(text)
        except ValueError:
            return None
    if has_percent_sign:
        return number / 100
    if percentage and abs(number) > 1:
        return number / 100
    return number


def _sequence(value: Any) -> list[Any] | None:
    if isinstance(value, (list, tuple, set)):
        return list(value)
    text = _clean_text(value)
    text = text.replace(";", ",")
    if text.startswith("{") and text.endswith("}"):
        text = "[" + text[1:-1] + "]"
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        stripped = text.strip("()[]{}")
        return [item.strip() for item in stripped.split(",") if item.strip()]
    return list(parsed) if isinstance(parsed, (list, tuple, set)) else None


UNIT_ALIASES = {
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "centimeter": "cm",
    "centimeters": "cm",
    "kilometer": "km",
    "kilometers": "km",
    "second": "s",
    "seconds": "s",
    "minute": "min",
    "minutes": "min",
    "hour": "h",
    "hours": "h",
    "degree": "deg",
    "degrees": "deg",
}


def _unit_value(value: Any) -> tuple[float, str] | None:
    match = re.fullmatch(
        r"\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:/\d+)?)\s*([A-Za-z°]+)\s*",
        _clean_text(value),
    )
    if not match:
        return None
    number = _number(match.group(1))
    unit = UNIT_ALIASES.get(match.group(2).lower(), match.group(2).lower())
    return (number, unit) if number is not None else None


def normalize_answer(
    value: Any,
    answer_type: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    answer_type = normalize_answer_type(answer_type)
    if answer_type not in ANSWER_TYPES:
        return {"success": False, "answer_type": answer_type, "value": None}
    normalized: Any
    if answer_type in {"integer", "decimal", "rational"}:
        normalized = _number(value)
    elif answer_type == "percentage":
        normalized = _number(value, percentage=True)
    elif answer_type == "boolean":
        lookup = {
            "yes": True,
            "true": True,
            "1": True,
            "\u662f": True,
            "no": False,
            "false": False,
            "0": False,
            "\u5426": False,
        }
        normalized = lookup.get(_clean_text(value).lower())
    elif answer_type in {
        "set",
        "unordered_collection",
    }:
        sequence = _sequence(value)
        normalized = (
            sorted(
                (
                    (
                        f"{numeric:.12g}"
                        if (numeric := _number(item)) is not None
                        else _clean_text(item)
                    )
                    for item in sequence
                ),
                key=str,
            )
            if sequence is not None
            else None
        )
    elif answer_type in {"ordered_tuple", "vector"}:
        sequence = _sequence(value)
        normalized = (
            [_clean_text(item) for item in sequence]
            if sequence is not None
            else None
        )
    elif answer_type == "matrix":
        sequence = _sequence(value)
        if sequence is not None and sequence and all(
            isinstance(row, (list, tuple)) for row in sequence
        ):
            width = len(sequence[0])
            normalized = (
                [[_number(item) for item in row] for row in sequence]
                if width and all(len(row) == width for row in sequence)
                else None
            )
        else:
            rows = [
                [item.strip() for item in row.split(",")]
                for row in _clean_text(value).strip("[]").split(";")
                if row.strip()
            ]
            normalized = (
                [[_number(item) for item in row] for row in rows]
                if rows and len({len(row) for row in rows}) == 1
                else None
            )
    elif answer_type == "unit_value":
        normalized = _unit_value(value)
    else:
        normalized = _clean_text(value).lower()
    return {
        "success": normalized is not None,
        "answer_type": answer_type,
        "value": normalized,
    }


def _numeric_equal(left: float, right: float, config: Mapping[str, Any]) -> bool:
    normalization = config["answer_normalization"]
    return math.isclose(
        left,
        right,
        rel_tol=float(normalization["relative_tolerance"]),
        abs_tol=float(normalization["absolute_tolerance"]),
    )


def _symbolic_equal(left: str, right: str) -> bool:
    try:
        import sympy

        return sympy.simplify(
            sympy.sympify(left.replace("^", "**"))
            - sympy.sympify(right.replace("^", "**"))
        ) == 0
    except Exception:
        return False


def _equation_equal(left: str, right: str) -> bool:
    try:
        import sympy

        def expression(text: str):
            lhs, rhs = text.split("=", 1)
            return sympy.expand(
                sympy.sympify(lhs.replace("^", "**"))
                - sympy.sympify(rhs.replace("^", "**"))
            )

        left_expr = expression(left)
        right_expr = expression(right)
        if sympy.simplify(left_expr - right_expr) == 0:
            return True
        ratio = sympy.simplify(left_expr / right_expr)
        return bool(ratio.is_number and ratio != 0)
    except Exception:
        return False


def _interval_signature(value: str) -> tuple[Any, Any, bool, bool] | None:
    text = _clean_text(value).replace("∞", "oo").replace(" ", "")
    match = re.fullmatch(r"([\[(])([^,]+),([^\])]+)([\])])", text)
    if not match:
        return None
    left_text, right_text = match.group(2), match.group(3)

    def endpoint(raw: str):
        if raw in {"-oo", "-inf", "-infinity"}:
            return -math.inf
        if raw in {"oo", "+oo", "inf", "+inf", "infinity"}:
            return math.inf
        return _number(raw)

    left, right = endpoint(left_text), endpoint(right_text)
    if left is None or right is None:
        return None
    return left, right, match.group(1) == "[", match.group(4) == "]"


def _inequality_set(value: str):
    import sympy

    text = _clean_text(value).replace("^", "**")
    relation = sympy.sympify(text)
    symbols = sorted(relation.free_symbols, key=lambda item: item.name)
    if len(symbols) != 1:
        return None
    return sympy.solve_univariate_inequality(
        relation,
        symbols[0],
        relational=False,
    )


def answers_equivalent(
    gold_answer: Any,
    predicted_answer: Any,
    answer_type: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    gold = normalize_answer(gold_answer, answer_type, config)
    predicted = normalize_answer(predicted_answer, answer_type, config)
    checks = {
        "answer_parse_success": gold["success"] and predicted["success"],
        "numeric_equivalence": False,
        "symbolic_equivalence": False,
        "unit_consistent": True,
        "format_valid": predicted["success"],
    }
    if not checks["answer_parse_success"]:
        return {
            "equivalent": False,
            "status": "ambiguous",
            "gold_normalized": gold,
            "predicted_normalized": predicted,
            "deterministic_checks": checks,
        }
    left, right = gold["value"], predicted["value"]
    equivalent = False
    if answer_type in {"integer", "decimal", "rational", "percentage"}:
        equivalent = _numeric_equal(float(left), float(right), config)
        checks["numeric_equivalence"] = equivalent
    elif answer_type == "matrix":
        equivalent = (
            len(left) == len(right)
            and all(len(a) == len(b) for a, b in zip(left, right))
            and all(
                _numeric_equal(float(a), float(b), config)
                for left_row, right_row in zip(left, right)
                for a, b in zip(left_row, right_row)
                if a is not None and b is not None
            )
            and all(
                a is not None and b is not None
                for left_row, right_row in zip(left, right)
                for a, b in zip(left_row, right_row)
            )
        )
        checks["numeric_equivalence"] = equivalent
    elif answer_type == "unit_value":
        checks["unit_consistent"] = left[1] == right[1]
        equivalent = checks["unit_consistent"] and _numeric_equal(
            float(left[0]), float(right[0]), config
        )
        checks["numeric_equivalence"] = equivalent
    elif answer_type == "symbolic_expression":
        equivalent = _symbolic_equal(str(left), str(right))
        checks["symbolic_equivalence"] = equivalent
    elif answer_type == "equation":
        equivalent = _equation_equal(str(left), str(right))
        checks["symbolic_equivalence"] = equivalent
    elif answer_type == "interval":
        left_interval = _interval_signature(str(left))
        right_interval = _interval_signature(str(right))
        equivalent = (
            left_interval is not None
            and right_interval is not None
            and left_interval == right_interval
        )
        checks["symbolic_equivalence"] = equivalent
    elif answer_type == "inequality":
        try:
            equivalent = _inequality_set(str(left)) == _inequality_set(str(right))
        except Exception:
            equivalent = str(left).replace(" ", "") == str(right).replace(" ", "")
        checks["symbolic_equivalence"] = equivalent
    else:
        equivalent = left == right
    return {
        "equivalent": bool(equivalent),
        "status": "correct" if equivalent else "incorrect",
        "gold_normalized": gold,
        "predicted_normalized": predicted,
        "deterministic_checks": checks,
    }


def attribute_error(
    record: Mapping[str, Any],
    parse_result: Mapping[str, Any],
    equivalence: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    status = str(parse_result.get("parse_status", "parse_failed"))
    evidence = []
    confidence = 1.0
    if status == "tool_violation":
        primary = "tool_violation"
    elif status == "prompt_echo":
        primary = "prompt_echo"
    elif status == "irrelevant_output":
        primary = "irrelevant_output"
    elif status != "success":
        primary = "parse_failed"
    elif equivalence.get("equivalent"):
        primary = None
    elif not equivalence.get("predicted_normalized", {}).get("success"):
        primary = "format_output_error"
    else:
        question = str(record.get("question", "")).lower()
        reasoning = " ".join(
            parse_result.get("parsed_response", {}).get("reasoning_summary", [])
        ).lower()
        if any(token in question for token in ("domain", "unit", "positive", "integer")):
            primary = "condition_missing"
            confidence = 0.78
        elif any(token in reasoning for token in ("formula", "identity", "theorem")):
            primary = "formula_memory_error"
            confidence = 0.76
        elif len(parse_result.get("parsed_response", {}).get("reasoning_summary", [])) >= 3:
            primary = "multi_step_logic_error"
            confidence = 0.74
        elif any(char.isdigit() for char in reasoning):
            primary = "calculation_error"
            confidence = 0.72
        else:
            primary = "concept_confusion"
            confidence = 0.65
        evidence.append(
            {
                "response_span": reasoning[:240],
                "reason": "Deterministic rule selected from output validity, answer equivalence, and response structure.",
            }
        )
    threshold = float(config["error_attribution"]["confidence_threshold"])
    needs_review = primary is not None and confidence < threshold
    if needs_review:
        primary = str(config["error_attribution"]["low_confidence_tag"])
    return {
        "is_correct": bool(equivalence.get("equivalent")),
        "primary_error_tag": primary,
        "secondary_error_tags": [],
        "evidence": evidence,
        "attribution_confidence": confidence,
        "needs_review": needs_review,
        "deterministic_checks": equivalence.get("deterministic_checks", {}),
    }
