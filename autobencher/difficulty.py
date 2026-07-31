"""Observable, auditable mathematics difficulty profiling.

Difficulty is treated as a property of the work required by a problem, not of
its subject label or of an LLM's unsupported impression.  The scalar 1--10
score is derived from five stored dimensions so every assignment can be
recomputed and inspected.
"""

from __future__ import annotations

import math
import re
from typing import Any, Mapping, Sequence


RUBRIC_VERSION = "observable_math_v1"

DEFAULT_WEIGHTS = {
    "reasoning_steps": 0.30,
    "operation_count": 0.20,
    "constraint_count": 0.20,
    "symbolic_depth": 0.20,
    "representation_load": 0.10,
}

DIMENSION_LIMITS = {
    "reasoning_steps": (1, 6),
    "operation_count": (0, 12),
    "constraint_count": (1, 6),
    "symbolic_depth": (0, 4),
    "representation_load": (0, 4),
}

DIFFICULTY_BANDS = (
    {
        "minimum": 1,
        "maximum": 2,
        "name": "foundational",
        "description": (
            "One concept, one or two direct transformations, little symbolic "
            "nesting, and no irrelevant arithmetic."
        ),
        "reasoning_steps": [1, 2],
        "operation_count": [1, 3],
        "constraint_count": [1, 1],
    },
    {
        "minimum": 3,
        "maximum": 4,
        "name": "routine_multistep",
        "description": (
            "A routine multi-step exercise with one main concept, up to two "
            "explicit constraints, and a short verification."
        ),
        "reasoning_steps": [2, 3],
        "operation_count": [3, 6],
        "constraint_count": [1, 2],
    },
    {
        "minimum": 5,
        "maximum": 6,
        "name": "integrated",
        "description": (
            "Several dependent transformations, multiple conditions or a "
            "nontrivial representation, while remaining a standard exercise."
        ),
        "reasoning_steps": [3, 5],
        "operation_count": [5, 9],
        "constraint_count": [2, 3],
    },
    {
        "minimum": 7,
        "maximum": 8,
        "name": "advanced",
        "description": (
            "Multiple concepts or cases, deeper symbolic structure, and "
            "substantial verification."
        ),
        "reasoning_steps": [4, 6],
        "operation_count": [8, 12],
        "constraint_count": [2, 5],
    },
    {
        "minimum": 9,
        "maximum": 10,
        "name": "expert",
        "description": (
            "Dense multi-concept reasoning, case analysis, or proof-like "
            "work. This band is outside the default production generation "
            "range."
        ),
        "reasoning_steps": [5, 6],
        "operation_count": [10, 12],
        "constraint_count": [3, 6],
    },
)

_MATH_FUNCTION = re.compile(
    r"\b(?:sin|cos|tan|cot|sec|csc|sqrt|exp|log|ln|Abs|factorial|"
    r"binomial|gcd|lcm|Mod|floor|ceiling|det|dot|mean|variance|"
    r"totient|divisor_count|isprime)\s*\(",
    flags=re.IGNORECASE,
)
_VARIABLE = re.compile(r"\b[a-zA-Z]\w*\b")
_OPERATOR = re.compile(r"(?<![eE])[+\-*/^]|(?:<=|>=|==|=|<|>)")
_DOMAIN_CONSTRAINT = re.compile(
    r"\b(?:positive|negative|nonzero|integer|integers|real|distinct|"
    r"at least|at most|greater than|less than|domain|range)\b",
    flags=re.IGNORECASE,
)


def _clamp(value: int, minimum: int, maximum: int) -> int:
    return max(minimum, min(maximum, int(value)))


def _band(score: int) -> dict[str, Any]:
    for item in DIFFICULTY_BANDS:
        if int(item["minimum"]) <= score <= int(item["maximum"]):
            return dict(item)
    return dict(DIFFICULTY_BANDS[-1])


def target_difficulty_profile(
    score: int,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    del config
    score = _clamp(score, 1, 10)
    band = _band(score)
    return {
        "rubric_version": RUBRIC_VERSION,
        "target_score": score,
        "band": band["name"],
        "description": band["description"],
        "expected_reasoning_steps": band["reasoning_steps"],
        "expected_operation_count": band["operation_count"],
        "expected_constraint_count": band["constraint_count"],
        "instruction": (
            "Match the mathematical work described by this profile. Do not "
            "inflate difficulty with large numbers, verbose wording, or "
            "unnecessary computation."
        ),
    }


def _details_view(details: Mapping[str, Any] | None) -> Mapping[str, Any]:
    details = details if isinstance(details, Mapping) else {}
    nested = details.get("deterministic_truth_solver")
    if isinstance(nested, Mapping):
        nested_details = nested.get("truth_validation_details")
        if isinstance(nested_details, Mapping):
            return nested_details
    return details


def _parenthesis_depth(text: str) -> int:
    depth = 0
    maximum = 0
    for char in text:
        if char in "([{":
            depth += 1
            maximum = max(maximum, depth)
        elif char in ")]}":
            depth = max(0, depth - 1)
    return maximum


def _dimension(
    name: str,
    value: int,
    evidence: Sequence[str],
) -> dict[str, Any]:
    minimum, maximum = DIMENSION_LIMITS[name]
    value = _clamp(value, minimum, maximum)
    normalized = (
        (value - minimum) / (maximum - minimum)
        if maximum > minimum
        else 0.0
    )
    return {
        "value": value,
        "minimum": minimum,
        "maximum": maximum,
        "normalized": round(normalized, 6),
        "evidence": list(evidence),
    }


def analyze_difficulty(
    question: str,
    answer_type: str = "text",
    truth_validation_details: Mapping[str, Any] | None = None,
    reasoning_steps: Sequence[Any] | None = None,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a five-dimensional profile and a reproducible 1--10 score."""
    question = " ".join(str(question or "").split())
    details = _details_view(truth_validation_details)
    expressions = details.get("normalized_expressions", [])
    if not isinstance(expressions, (list, tuple)):
        expressions = [expressions] if expressions else []
    expression_text = " ; ".join(str(item) for item in expressions)
    analysis_text = expression_text or question
    operator_text = re.sub(
        r"(?<=[A-Za-z])-(?=[A-Za-z])",
        "",
        analysis_text,
    )

    steps = [
        str(step).strip()
        for step in (reasoning_steps or [])
        if str(step).strip()
    ]
    operation_count = len(_OPERATOR.findall(operator_text))
    operation_count += len(_MATH_FUNCTION.findall(operator_text))

    explicit_constraints = max(
        1,
        len(expressions),
        len(re.findall(r"(?:<=|>=|==|=|<|>)", analysis_text)),
    )
    domain_constraints = len(_DOMAIN_CONSTRAINT.findall(question))
    constraint_count = explicit_constraints + min(2, domain_constraints)

    variables = {
        token
        for token in _VARIABLE.findall(analysis_text)
        if token.lower()
        not in {
            "compute",
            "solve",
            "for",
            "with",
            "respect",
            "to",
            "from",
            "find",
            "the",
            "limit",
            "integrate",
            "differentiate",
        }
        and not _MATH_FUNCTION.fullmatch(token + "(")
    }
    symbolic_depth = 0
    if variables:
        symbolic_depth += 1
    if len(variables) >= 2:
        symbolic_depth += 1
    if _MATH_FUNCTION.search(analysis_text) or "^" in analysis_text:
        symbolic_depth += 1
    if _parenthesis_depth(analysis_text) >= 2:
        symbolic_depth += 1

    representation_features = []
    lowered = question.lower()
    if len(question.split()) >= 24:
        representation_features.append("word_problem_translation")
    if re.search(r"(?:%|percent|ratio|per\s+\w+)", lowered):
        representation_features.append("ratio_or_rate")
    if re.search(
        r"\b(?:meter|metre|mile|km|kilometer|hour|minute|second|"
        r"dollar|cost|price|area|volume|degree)\w*\b",
        lowered,
    ):
        representation_features.append("units_or_measurement")
    if re.search(r"\b(?:matrix|vector|coordinate|triangle|circle)\b", lowered):
        representation_features.append("structured_representation")
    if re.search(r"\b(?:case|piecewise|otherwise|either)\b", lowered):
        representation_features.append("case_distinction")
    if str(answer_type) in {"matrix", "vector", "interval", "inequality"}:
        representation_features.append(f"answer_type:{answer_type}")

    estimated_steps = max(
        1,
        int(math.ceil(
            (
                min(operation_count, 12)
                + min(constraint_count, 6)
                + min(symbolic_depth, 4)
            )
            / 4.0
        )),
    )
    reasoning_count = (
        max(len(steps), estimated_steps)
        if steps
        else estimated_steps
    )

    dimensions = {
        "reasoning_steps": _dimension(
            "reasoning_steps",
            reasoning_count,
            (
                [
                    f"verified_reasoning_summary_count={len(steps)}",
                    f"observable_structural_lower_bound={estimated_steps}",
                ]
                if steps
                else ["estimated_from_observable_structure"]
            ),
        ),
        "operation_count": _dimension(
            "operation_count",
            operation_count,
            [f"parsed_operation_or_function_count={operation_count}"],
        ),
        "constraint_count": _dimension(
            "constraint_count",
            constraint_count,
            [
                f"explicit_constraints={explicit_constraints}",
                f"domain_constraints={domain_constraints}",
            ],
        ),
        "symbolic_depth": _dimension(
            "symbolic_depth",
            symbolic_depth,
            [
                f"distinct_symbol_tokens={len(variables)}",
                f"parenthesis_depth={_parenthesis_depth(analysis_text)}",
            ],
        ),
        "representation_load": _dimension(
            "representation_load",
            len(set(representation_features)),
            sorted(set(representation_features))
            or ["direct_symbolic_or_numeric_representation"],
        ),
    }

    difficulty_config = (
        config.get("difficulty", {})
        if isinstance(config, Mapping)
        else {}
    )
    configured_weights = difficulty_config.get(
        "dimension_weights",
        DEFAULT_WEIGHTS,
    )
    weights = {
        name: float(configured_weights.get(name, DEFAULT_WEIGHTS[name]))
        for name in DEFAULT_WEIGHTS
    }
    weight_total = sum(weights.values())
    if weight_total <= 0:
        raise ValueError("difficulty dimension weights must sum to a positive value")
    weights = {
        name: value / weight_total
        for name, value in weights.items()
    }
    weighted = sum(
        weights[name] * float(dimensions[name]["normalized"])
        for name in DEFAULT_WEIGHTS
    )
    score = _clamp(int(math.floor(1.0 + 9.0 * weighted + 0.5)), 1, 10)
    confidence = 0.60
    if expressions:
        confidence += 0.15
    if steps:
        confidence += 0.20
    confidence = min(0.95, confidence)
    return {
        "rubric_version": str(
            difficulty_config.get("rubric_version", RUBRIC_VERSION)
        ),
        "scoring_method": "weighted_observable_dimensions",
        "score": score,
        "band": _band(score)["name"],
        "confidence": round(confidence, 3),
        "dimensions": dimensions,
        "dimension_weights": {
            name: round(value, 6)
            for name, value in weights.items()
        },
        "anti_shortcut_policy": (
            "Large numbers, verbose wording, and answer rarity do not "
            "independently increase difficulty."
        ),
    }


def assess_difficulty(
    question: str,
    answer_type: str,
    truth_validation_details: Mapping[str, Any] | None,
    reasoning_steps: Sequence[Any] | None,
    requested_score: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Attach target agreement, bounds, and the score used by the sampler."""
    profile = analyze_difficulty(
        question,
        answer_type,
        truth_validation_details,
        reasoning_steps,
        config,
    )
    difficulty_config = config["difficulty"]
    requested = _clamp(requested_score, 1, 10)
    observed = int(profile["score"])
    tolerance = int(difficulty_config["target_tolerance"])
    minimum_confidence = float(
        difficulty_config["minimum_profile_confidence"]
    )
    trusted = float(profile["confidence"]) >= minimum_confidence
    study = config.get("study", {})
    components = (
        study.get("components", {})
        if isinstance(study, Mapping)
        else {}
    )
    use_observed = bool(
        difficulty_config["use_observed_score_for_adaptive_sampling"]
    ) and bool(components.get("observed_difficulty", True))
    if isinstance(study, Mapping) and study.get("variant") == (
        "full_no_observed_difficulty"
    ):
        use_observed = False
    effective = observed if trusted and use_observed else requested
    lower = int(config["generation"]["minimum_difficulty"])
    upper = int(config["generation"]["maximum_difficulty"])
    profile.update(
        {
            "requested_score": requested,
            "target_profile": target_difficulty_profile(
                requested,
                config,
            ),
            "score_gap": observed - requested,
            "absolute_score_gap": abs(observed - requested),
            "target_tolerance": tolerance,
            "within_target_tolerance": abs(observed - requested) <= tolerance,
            "within_generation_bounds": lower <= observed <= upper,
            "profile_trusted": trusted,
            "effective_score": effective,
            "effective_score_source": (
                "observed"
                if trusted and use_observed
                else "requested"
            ),
            "mismatch_action": str(
                difficulty_config["mismatch_action"]
            ),
        }
    )
    return profile
