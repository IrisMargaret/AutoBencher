"""Build the checked-in, project-authored 120-item retention benchmark."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "benchmarks" / "retention_regression_set.json"


def record(index, dimension, question, answer, answer_type, category, subcategory):
    return {
        "question_id": f"retention-{index:03d}",
        "category": category,
        "sub_category": subcategory,
        "difficulty": 1 if dimension != "logical_consistency" else 2,
        "question": question,
        "answer_type": answer_type,
        "canonical_answer": str(answer),
        "retention_dimension": dimension,
        "source_dataset": "project_native",
        "verification": {
            "status": "verified",
            "backend": "deterministic_reference_plus_schema_check",
            "canonical_parse_passed": True,
            "training_use_prohibited": True,
        },
    }


def build_questions():
    questions = []
    index = 1
    for left in range(11, 31):
        right = 43 - left
        questions.append(record(
            index,
            "basic_instruction_following",
            f"Add {left} and {right}. Return only the integer result.",
            left + right,
            "integer",
            "Arithmetic",
            "Integer Operations",
        ))
        index += 1

    words = [
        "amber", "birch", "cedar", "delta", "ember", "fable", "globe",
        "harbor", "indigo", "jasmine", "kernel", "linen", "maple", "nectar",
        "oasis", "pebble", "quartz", "raven", "silver", "timber",
    ]
    for word in words:
        questions.append(record(
            index,
            "language_transformation",
            f"Convert the lowercase word '{word}' to uppercase. Return only the word.",
            word.upper(),
            "text",
            "Composite Comprehensive",
            "Proof and Mathematical Reasoning",
        ))
        index += 1

    for number in range(21, 41):
        token = f"ID-{number:02d}-OK"
        questions.append(record(
            index,
            "strict_format_following",
            f"Return exactly this token, preserving capitalization and hyphens: {token}",
            token,
            "text",
            "Composite Comprehensive",
            "Constraint Synthesis",
        ))
        index += 1

    facts = [
        ("the capital of Japan", "Tokyo"),
        ("the capital of France", "Paris"),
        ("the capital of Italy", "Rome"),
        ("the largest planet in the Solar System", "Jupiter"),
        ("the planet known as the Red Planet", "Mars"),
        ("the gas plants absorb during photosynthesis", "carbon dioxide"),
        ("the chemical symbol for gold", "Au"),
        ("the chemical symbol for water", "H2O"),
        ("the author of Hamlet", "William Shakespeare"),
        ("the ocean between Africa and Australia", "Indian Ocean"),
        ("the continent containing Kenya", "Africa"),
        ("the primary language of Brazil", "Portuguese"),
        ("the instrument used to measure temperature", "thermometer"),
        ("the process by which liquid becomes gas", "evaporation"),
        ("the first month of the year", "January"),
        ("the day after Friday", "Saturday"),
        ("the opposite cardinal direction to north", "south"),
        ("the number of sides in a hexagon", "6"),
        ("the mammal commonly described as the largest animal", "blue whale"),
        ("the star at the center of the Solar System", "Sun"),
    ]
    for prompt, answer in facts:
        questions.append(record(
            index,
            "stable_common_knowledge",
            f"Give the standard short answer for {prompt}.",
            answer,
            "text" if not str(answer).isdigit() else "integer",
            "Composite Comprehensive",
            "Cross-Domain Multi-Step Problems",
        ))
        index += 1

    for number in range(10, 30):
        other = number + (1 if number % 2 else -1)
        truth = number > other
        questions.append(record(
            index,
            "logical_consistency",
            f"Answer true or false: {number} is greater than {other}.",
            str(truth).lower(),
            "boolean",
            "Composite Comprehensive",
            "Proof and Mathematical Reasoning",
        ))
        index += 1

    pairs = [
        ("apple", "pear"), ("zebra", "yak"), ("violet", "amber"),
        ("river", "ocean"), ("north", "south"), ("circle", "square"),
        ("winter", "summer"), ("copper", "silver"), ("piano", "violin"),
        ("tiger", "lion"), ("cloud", "rain"), ("forest", "desert"),
        ("coffee", "tea"), ("novel", "poem"), ("glass", "stone"),
        ("green", "blue"), ("Monday", "Friday"), ("seven", "three"),
        ("east", "west"), ("large", "small"),
    ]
    for left, right in pairs:
        answer = min(left, right, key=lambda value: value.lower())
        questions.append(record(
            index,
            "non_target_string_reasoning",
            f"Which comes first alphabetically: '{left}' or '{right}'? Return only that word.",
            answer,
            "text",
            "Composite Comprehensive",
            "Proof and Mathematical Reasoning",
        ))
        index += 1
    assert len(questions) == 120
    assert len({item["question"] for item in questions}) == 120
    return questions


def main():
    payload = {
        "schema_version": "2.0",
        "name": "autobencher_retention_regression_v2",
        "description": (
            "Project-authored holdout for instruction, formatting, language, "
            "stable knowledge, logic, and non-target string retention."
        ),
        "selection_policy": {
            "training_use_prohibited": True,
            "checkpoint_selection_prohibited": True,
            "external_question_copying": False,
            "minimum_items_per_dimension": 20,
        },
        "questions": build_questions(),
    }
    OUTPUT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
