"""Deterministically build the project-native 81-question fixed benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from autobencher.experiment import atomic_json
from autobencher.fixed_benchmark import load_fixed_test_set
from autobencher.structured import normalize_answer
from autobencher.structured import normalize_generated_gold_contract


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_ROOT / "benchmarks" / "fixed_math_test_set.json"

DESIGN_REFERENCES = [
    {
        "name": "GSM8K",
        "url": "https://github.com/openai/grade-school-math",
        "role": "multi-step arithmetic and word-problem design reference",
        "questions_copied": False,
    },
    {
        "name": "MATH",
        "url": "https://github.com/hendrycks/math",
        "role": "subject coverage and symbolic-answer design reference",
        "questions_copied": False,
    },
    {
        "name": "MMLU",
        "url": "https://github.com/hendrycks/test",
        "role": "broad mathematical competency design reference",
        "questions_copied": False,
    },
    {
        "name": "DeepMind Mathematics Dataset",
        "url": (
            "https://github.com/google-deepmind/mathematics_dataset"
        ),
        "role": "parameterized question generation design reference",
        "questions_copied": False,
    },
]


def _extra_specs() -> list[tuple[str, str, int, str, str, str]]:
    """Return two independently authored extensions for every subcategory."""
    return [
        ("Arithmetic", "Integer Operations", 3,
         "Compute (-24)/6 + 5*(7 - 3).", "integer", "16"),
        ("Arithmetic", "Integer Operations", 4,
         "Compute |-17| - 3^2 + 2*(-4).", "integer", "0"),
        ("Arithmetic", "Fraction and Decimal Operations", 3,
         "Compute 7/12 - 5/18 as a reduced fraction.", "rational", "11/36"),
        ("Arithmetic", "Fraction and Decimal Operations", 4,
         "Compute 1.25 + 3/8 - 0.4 as a reduced fraction.",
         "rational", "49/40"),
        ("Arithmetic", "Ratio and Percentage", 3,
         "Red and blue beads are in the ratio 3:5. If there are 64 beads, "
         "how many are red?", "integer", "24"),
        ("Arithmetic", "Ratio and Percentage", 4,
         "A price of 250 dollars is increased by 20% and then reduced by "
         "10%. What is the final price?", "decimal", "270"),
        ("Algebra", "Linear Equations", 3,
         "Solve for x: 3*(2*x - 5) + 4 = 5*x + 9.",
         "integer", "20"),
        ("Algebra", "Linear Equations", 5,
         "Solve for x: (x - 2)/3 + (x + 1)/2 = 7.",
         "rational", "43/5"),
        ("Algebra", "Systems of Equations", 3,
         "Solve for (x, y): 3*x + 2*y = 16 and x - y = 2.",
         "ordered_tuple", "(4, 2)"),
        ("Algebra", "Systems of Equations", 5,
         "Solve for (x, y, z): x + y + z = 9, 2*x - y + z = 8, "
         "and x + 2*y - z = 3.", "ordered_tuple", "(3, 2, 4)"),
        ("Algebra", "Polynomials and Inequalities", 4,
         "Solve the inequality x^2 + x - 12 < 0 over the real numbers.",
         "interval", "(-4, 3)"),
        ("Algebra", "Polynomials and Inequalities", 5,
         "Find all real roots of x^3 - 4*x = 0.",
         "set", "{-2, 0, 2}"),
        ("Geometry & Trigonometry", "Plane Geometry", 3,
         "A triangle has base 12 and perpendicular height 7. Find its area.",
         "decimal", "42"),
        ("Geometry & Trigonometry", "Plane Geometry", 5,
         "Find the exact area of a 120-degree sector of a circle of radius 6.",
         "symbolic_expression", "12*pi"),
        ("Geometry & Trigonometry", "Solid Geometry", 4,
         "Find the exact space diagonal of a rectangular prism with side "
         "lengths 4, 5, and 9.", "symbolic_expression", "sqrt(122)"),
        ("Geometry & Trigonometry", "Solid Geometry", 5,
         "Find the exact surface area of a sphere of radius 3.",
         "symbolic_expression", "36*pi"),
        ("Geometry & Trigonometry", "Trigonometric Reasoning", 3,
         "In a right triangle, the side adjacent to theta is 12 and the "
         "hypotenuse is 13. Find cos(theta).", "rational", "12/13"),
        ("Geometry & Trigonometry", "Trigonometric Reasoning", 5,
         "Solve 2*sin(x) = 1 for x in the interval [0, 2*pi).",
         "set", "{pi/6, 5*pi/6}"),
        ("Probability & Statistics", "Basic Probability", 3,
         "A bag contains 5 red, 3 blue, and 2 green balls. One ball is "
         "drawn uniformly. What is the probability it is not red?",
         "rational", "1/2"),
        ("Probability & Statistics", "Basic Probability", 4,
         "Two fair six-sided dice are rolled. What is the probability that "
         "their sum is 9?", "rational", "1/9"),
        ("Probability & Statistics", "Combinatorics", 3,
         "In how many orders can 6 distinct books be arranged on a shelf?",
         "integer", "720"),
        ("Probability & Statistics", "Combinatorics", 5,
         "How many 3-person committees can be chosen from 10 people if two "
         "specified people may not both serve?", "integer", "112"),
        ("Probability & Statistics", "Descriptive Statistics", 3,
         "Find the median of 3, 5, 7, 11, and 14.", "decimal", "7"),
        ("Probability & Statistics", "Descriptive Statistics", 4,
         "Scores 70, 80, and 95 have frequencies 2, 3, and 1. Find the "
         "weighted mean as a reduced fraction.", "rational", "475/6"),
        ("Word Problems", "Rate and Distance", 3,
         "A cyclist rides at 18 km/h for 1.75 hours. How far does the cyclist "
         "travel in kilometers?", "decimal", "31.5"),
        ("Word Problems", "Rate and Distance", 4,
         "Two trains 400 km apart move toward each other at 72 km/h and "
         "88 km/h. After how many hours do they meet?", "decimal", "2.5"),
        ("Word Problems", "Work and Mixture", 4,
         "One machine completes a job in 4 hours and another in 6 hours. "
         "How many hours do they need working together?",
         "rational", "12/5"),
        ("Word Problems", "Work and Mixture", 5,
         "How many liters of 50% solution must be added to 10 liters of 20% "
         "solution to obtain a 32% solution?", "rational", "20/3"),
        ("Word Problems", "Financial Applications", 3,
         "What interest is earned when 1000 dollars is compounded annually "
         "at 10% for 2 years?", "decimal", "210"),
        ("Word Problems", "Financial Applications", 4,
         "A product has fixed cost 600 dollars, variable cost 8 dollars per "
         "unit, and selling price 20 dollars. How many units break even?",
         "integer", "50"),
        ("Number Theory", "Divisibility and Factors", 3,
         "Find the least common multiple of 18 and 24.", "integer", "72"),
        ("Number Theory", "Divisibility and Factors", 5,
         "How many positive divisors does 2^3 * 3^2 * 5 have?",
         "integer", "24"),
        ("Number Theory", "Prime Factorization", 3,
         "Write 756 as a product of prime powers.",
         "symbolic_expression", "2^2*3^3*7"),
        ("Number Theory", "Prime Factorization", 5,
         "Find the least positive integer divisible by both 84 and 90.",
         "integer", "1260"),
        ("Number Theory", "Modular Arithmetic", 4,
         "Find the remainder when 3^100 is divided by 7.", "integer", "4"),
        ("Number Theory", "Modular Arithmetic", 5,
         "Find the least nonnegative integer x satisfying 5*x = 3 (mod 11).",
         "integer", "5"),
        ("Calculus", "Limits and Continuity", 3,
         "Find the limit of (x^2 - 16)/(x - 4) as x approaches 4.",
         "integer", "8"),
        ("Calculus", "Limits and Continuity", 5,
         "Find the limit of sin(3*x)/x as x approaches 0.",
         "integer", "3"),
        ("Calculus", "Differentiation", 4,
         "Differentiate x^4 - 3*x^2 + 5 with respect to x.",
         "symbolic_expression", "4*x^3 - 6*x"),
        ("Calculus", "Differentiation", 5,
         "Differentiate (x^2 + 1)*exp(x) with respect to x.",
         "symbolic_expression", "(x^2 + 2*x + 1)*exp(x)"),
        ("Calculus", "Integration", 4,
         "Evaluate the integral of 2*x + 1 from x = 0 to x = 3.",
         "integer", "12"),
        ("Calculus", "Integration", 5,
         "Evaluate the integral of sin(x) from x = 0 to x = pi.",
         "integer", "2"),
        ("Linear Algebra", "Matrix Operations", 4,
         "Compute A^2 for A = [[2, 1], [1, 2]].",
         "matrix", "[[5, 4], [4, 5]]"),
        ("Linear Algebra", "Matrix Operations", 5,
         "Find the inverse of [[2, 1], [5, 3]].",
         "matrix", "[[3, -1], [-5, 2]]"),
        ("Linear Algebra", "Linear Systems", 4,
         "Solve for (x, y, z): 2*x + y - z = 1, x - y + 2*z = 5, "
         "and 3*x + 2*y + z = 10.", "ordered_tuple", "(1, 2, 3)"),
        ("Linear Algebra", "Linear Systems", 5,
         "Solve for (x, y, z): x + y + z = 3, x + 2*y + 3*z = 6, "
         "and 2*x - y + z = 2.", "ordered_tuple", "(1, 1, 1)"),
        ("Linear Algebra", "Vectors and Vector Spaces", 4,
         "Find the cross product (1, 2, 3) x (2, -1, 1).",
         "vector", "[5, 5, -5]"),
        ("Linear Algebra", "Vectors and Vector Spaces", 4,
         "Find the Euclidean norm of the vector (3, 4, 12).",
         "integer", "13"),
        ("Composite Comprehensive", "Cross-Domain Multi-Step Problems", 4,
         "A circle has area 49*pi. Find its exact circumference.",
         "symbolic_expression", "14*pi"),
        ("Composite Comprehensive", "Cross-Domain Multi-Step Problems", 5,
         "An item priced at 150 dollars receives a 20% discount and then 8% "
         "sales tax. What is the final price?", "decimal", "129.6"),
        ("Composite Comprehensive", "Proof and Mathematical Reasoning", 3,
         "True or false: the square of every odd integer is odd.",
         "boolean", "true"),
        ("Composite Comprehensive", "Proof and Mathematical Reasoning", 5,
         "True or false: if a*b is divisible by 6, then a or b must be "
         "divisible by 6.", "boolean", "false"),
        ("Composite Comprehensive", "Constraint Synthesis", 4,
         "Find the least nonnegative integer n such that n leaves remainder "
         "2 modulo 3 and remainder 3 modulo 5.", "integer", "8"),
        ("Composite Comprehensive", "Constraint Synthesis", 5,
         "How many ordered pairs (x, y) of nonnegative integers satisfy "
         "x + y = 10 and x is even?", "integer", "6"),
    ]


def build_payload(base_path: Path) -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    originals = [
        dict(item)
        for item in base["questions"]
        if int(str(item["question_id"]).split("-")[-1]) <= 27
    ]
    for item in originals:
        item["source_dataset"] = "project_native"
        item["construction_method"] = "human_curated_original_v3"
        item.update(
            normalize_generated_gold_contract(
                item["question"],
                item["canonical_answer"],
                item["answer_type"],
                item.get("tolerance"),
            )
        )
        parsed = normalize_answer(
            item["canonical_answer"],
            item["answer_type"],
            {},
        )
        if not parsed["success"]:
            raise ValueError(
                f"Canonical answer for {item['question_id']} does not parse"
            )
        item["verification"] = {
            "backend": "human_curated_plus_typed_parser",
            "canonical_parse_passed": True,
            "question_sha256": hashlib.sha256(
                item["question"].encode("utf-8")
            ).hexdigest(),
        }

    questions = list(originals)
    for index, spec in enumerate(_extra_specs(), start=28):
        category, subcategory, difficulty, question, answer_type, answer = spec
        contract = normalize_generated_gold_contract(
            question,
            answer,
            answer_type,
        )
        answer_type = contract["answer_type"]
        answer = contract["canonical_answer"]
        parsed = normalize_answer(answer, answer_type, {})
        if not parsed["success"]:
            raise ValueError(
                f"Canonical answer for project-fixed-{index:03d} "
                f"does not parse: {parsed}"
            )
        questions.append(
            {
                "question_id": f"project-fixed-{index:03d}",
                "category": category,
                "sub_category": subcategory,
                "difficulty": difficulty,
                "question": question,
                "answer_type": answer_type,
                "canonical_answer": answer,
                "display_answer": contract["display_answer"],
                "tolerance": contract["tolerance"],
                "exact_canonical_answer": contract[
                    "exact_canonical_answer"
                ],
                "source_dataset": "project_native",
                "construction_method": "human_curated_original_v3",
                "verification": {
                    "backend": "human_curated_plus_typed_parser",
                    "canonical_parse_passed": True,
                    "question_sha256": hashlib.sha256(
                        question.encode("utf-8")
                    ).hexdigest(),
                },
            }
        )
    return {
        "schema_version": "2.0",
        "name": "autobencher_project_native_fixed_math_v3",
        "description": (
            "Immutable 81-question project-native holdout with three questions "
            "for every configured mathematics subcategory."
        ),
        "license": "Project-authored benchmark; no external question copied.",
        "design_references": DESIGN_REFERENCES,
        "selection_policy": {
            "questions_per_subcategory": 3,
            "difficulty_intent": "basic, intermediate, and advanced coverage",
            "external_question_copying": False,
            "training_use_prohibited": True,
        },
        "questions": questions,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = Path(args.output).expanduser().resolve()
    payload = build_payload(DEFAULT_OUTPUT)
    atomic_json(payload, output)
    config = {
        "fixed_test": {
            "dataset_path": output.as_posix(),
            "require_all_subcategories": True,
        }
    }
    questions, metadata = load_fixed_test_set(config)
    if len(questions) != 81:
        raise RuntimeError(f"Expected 81 questions, found {len(questions)}")
    print(json.dumps(metadata, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
