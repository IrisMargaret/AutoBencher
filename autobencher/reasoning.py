"""Validation for evaluator-authored, training-safe mathematical derivations."""

from __future__ import annotations

import re
from typing import Any, Mapping


_FORBIDDEN_MARKERS = (
    "```",
    "<|",
    "|>",
    "system:",
    "assistant:",
    "user:",
    "human:",
    "question_json",
    "python_code",
    "ignore previous",
    "browse",
    "search the web",
    "tool call",
)

_VAGUE_STEPS = {
    "analyze the problem",
    "calculate the answer",
    "check the answer",
    "compute",
    "compute the answer",
    "find the answer",
    "solve",
    "solve the equation",
    "solve the problem",
    "substitute",
    "verify",
    "verify the answer",
}

_CONCRETE_MATH = re.compile(
    r"""
    \d
    |[=<>+\-*/^%()[\]{}]
    |\b(?:
        add|subtract|multiply|divide|factor|expand|simplif|substitut|
        equation|inequalit|constraint|derivative|differentiat|integral|
        integrat|limit|matrix|determinant|vector|probability|combination|
        permutation|ratio|percent|remainder|modulo|root|logarithm|angle|
        radius|diameter|area|volume|distance|rate|time|mean|median|
        variance|therefore|thus|hence
    )\w*
    """,
    re.IGNORECASE | re.VERBOSE,
)

_VERIFICATION_LANGUAGE = re.compile(
    r"\b(?:check|verify|substitut|recompute|satisf|hold|original|both|"
    r"independent|confirm)\w*",
    re.IGNORECASE,
)


def _answer_atoms(answer: Any) -> tuple[list[str], list[str]]:
    text = str(answer or "")
    numeric = list(
        dict.fromkeys(
            token.lstrip("+")
            for token in re.findall(r"[+-]?\d+(?:\.\d+)?", text)
        )
    )
    symbolic = [
        token.lower()
        for token in dict.fromkeys(re.findall(r"[A-Za-z]+", text))
        if token.lower()
        not in {
            "and",
            "or",
            "log",
            "ln",
            "sqrt",
            "matrix",
            "set",
        }
    ]
    return numeric, symbolic


def validate_training_reasoning(
    raw_steps: Any,
    config: Mapping[str, Any],
    *,
    verified_answer: Any | None = None,
) -> tuple[list[str], str | None]:
    """Return normalized concrete steps or a stable training rejection reason."""
    dataset_config = config["dataset"]
    if not isinstance(raw_steps, list):
        return [], "missing_gold_reasoning_steps"
    steps = [
        re.sub(r"\s+", " ", str(step or "")).strip()
        for step in raw_steps
    ]
    minimum = int(dataset_config["min_gold_reasoning_steps"])
    maximum = int(dataset_config["max_gold_reasoning_steps"])
    minimum_chars = int(
        dataset_config.get("min_gold_reasoning_chars_per_step", 1)
    )
    maximum_chars = int(
        dataset_config["max_gold_reasoning_chars_per_step"]
    )
    if not minimum <= len(steps) <= maximum or any(not step for step in steps):
        return [], "invalid_gold_reasoning_steps"
    if any(
        len(step) > maximum_chars
        or any(marker in step.lower() for marker in _FORBIDDEN_MARKERS)
        for step in steps
    ):
        return [], "unsafe_gold_reasoning_steps"

    if bool(dataset_config.get("require_concrete_gold_reasoning", False)):
        for step in steps:
            normalized = step.lower().strip(" .:;")
            if len(step) < minimum_chars or normalized in _VAGUE_STEPS:
                return [], "non_concrete_gold_reasoning_steps"
        concrete_count = sum(bool(_CONCRETE_MATH.search(step)) for step in steps)
        if concrete_count < min(2, len(steps)):
            return [], "non_concrete_gold_reasoning_steps"

    joined = " ".join(steps)
    if bool(
        dataset_config.get("require_gold_reasoning_verification", False)
    ) and not _VERIFICATION_LANGUAGE.search(joined):
        return [], "missing_gold_reasoning_verification"

    if (
        verified_answer is not None
        and bool(
            dataset_config.get("require_gold_answer_in_reasoning", False)
        )
    ):
        numeric_atoms, symbolic_atoms = _answer_atoms(verified_answer)
        normalized_joined = joined.lower()
        numeric_matches = sum(
            bool(
                re.search(
                    rf"(?<!\d){re.escape(atom)}(?!\d)",
                    joined,
                )
            )
            for atom in numeric_atoms
        )
        if numeric_atoms and numeric_matches < min(2, len(numeric_atoms)):
            return [], "gold_reasoning_not_anchored_to_answer"
        if (
            not numeric_atoms
            and symbolic_atoms
            and not any(atom in normalized_joined for atom in symbolic_atoms)
        ):
            return [], "gold_reasoning_not_anchored_to_answer"

    return steps, None
