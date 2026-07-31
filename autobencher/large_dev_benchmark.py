"""Deterministic 540-item development stress set (never a blind set)."""

from __future__ import annotations

import hashlib
import math
import os
from fractions import Fraction
from pathlib import Path
from typing import Any

from autobencher.config import DEFAULT_TAXONOMY
from autobencher.experiment import atomic_json
from autobencher.structured import normalize_generated_gold_contract


def _answer(value: Any) -> str:
    if isinstance(value, Fraction):
        return f"{value.numerator}/{value.denominator}"
    return str(value).lower() if isinstance(value, bool) else str(value)


def _item(category: str, subcategory: str, index: int) -> tuple[str, str, str]:
    i = index + 2
    key = (category, subcategory)
    if key == ("Arithmetic", "Integer Operations"):
        return f"Compute {i}*{i + 3} - {2 * i}.", "integer", _answer(i * (i + 3) - 2 * i)
    if key == ("Arithmetic", "Fraction and Decimal Operations"):
        value = Fraction(i, i + 1) + Fraction(i + 2, i + 3)
        return f"Compute {i}/{i + 1} + {i + 2}/{i + 3} as a reduced fraction.", "rational", _answer(value)
    if key == ("Arithmetic", "Ratio and Percentage"):
        return f"A quantity of {10 * i} is increased by {i}%. Give the exact result.", "rational", _answer(Fraction(10 * i * (100 + i), 100))
    if key == ("Algebra", "Linear Equations"):
        x, a, b = i + 1, i % 5 + 2, i - 1
        return f"Solve for x: {a}*x + {b} = {a * x + b}.", "integer", _answer(x)
    if key == ("Algebra", "Systems of Equations"):
        x, y = i, i + 2
        return f"Solve for (x, y): x + y = {x + y} and 2*x - y = {2*x - y}.", "ordered_tuple", f"({x}, {y})"
    if key == ("Algebra", "Polynomials and Inequalities"):
        a, b = i, i + 2
        return f"Find all real roots of (x - {a})*(x - {b}) = 0.", "set", f"{{{a}, {b}}}"
    if key == ("Geometry & Trigonometry", "Plane Geometry"):
        return f"Find the area of a rectangle with side lengths {i + 2} and {i + 5}.", "integer", _answer((i + 2) * (i + 5))
    if key == ("Geometry & Trigonometry", "Solid Geometry"):
        return f"Find the volume of a rectangular prism with sides {i}, {i + 1}, and {i + 3}.", "integer", _answer(i * (i + 1) * (i + 3))
    if key == ("Geometry & Trigonometry", "Trigonometric Reasoning"):
        scale = i
        return f"A right triangle has adjacent side {3 * scale} and hypotenuse {5 * scale}. Find cos(theta).", "rational", "3/5"
    if key == ("Probability & Statistics", "Basic Probability"):
        return f"A bag has {i} red and {i + 3} blue balls. Find the probability of drawing red.", "rational", _answer(Fraction(i, 2 * i + 3))
    if key == ("Probability & Statistics", "Combinatorics"):
        n = i + 4
        return f"How many unordered pairs can be selected from {n} distinct objects?", "integer", _answer(math.comb(n, 2))
    if key == ("Probability & Statistics", "Descriptive Statistics"):
        values = [i, i + 2, i + 5, i + 9]
        return f"Find the arithmetic mean of {', '.join(map(str, values))} as a reduced fraction.", "rational", _answer(Fraction(sum(values), 4))
    if key == ("Word Problems", "Rate and Distance"):
        speed, hours = i + 10, Fraction(i + 1, 2)
        return f"A vehicle travels at {speed} km/h for {hours.numerator}/{hours.denominator} hours. Find the distance.", "rational", _answer(speed * hours)
    if key == ("Word Problems", "Work and Mixture"):
        a, b = i + 2, i + 5
        return f"One machine finishes a job in {a} hours and another in {b} hours. Find their combined time.", "rational", _answer(Fraction(a * b, a + b))
    if key == ("Word Problems", "Financial Applications"):
        principal, rate = 100 * i, i + 1
        return f"Find the simple interest on {principal} dollars at {rate}% per year for 2 years.", "rational", _answer(Fraction(principal * rate * 2, 100))
    if key == ("Number Theory", "Divisibility and Factors"):
        a, b = 2 * i, 3 * (i + 1)
        return f"Find gcd({a}, {b}).", "integer", _answer(math.gcd(a, b))
    if key == ("Number Theory", "Prime Factorization"):
        a = index % 4 + 1
        b = (index // 4) % 3 + 1
        c = (index // 12) % 2 + 1
        value = (2**a) * (3**b) * (5**c)
        return f"Write {value} as a product of prime powers.", "symbolic_expression", f"2^{a}*3^{b}*5^{c}"
    if key == ("Number Theory", "Modular Arithmetic"):
        exponent = i + 4
        return f"Find the remainder when 2^{exponent} is divided by 7.", "integer", _answer(pow(2, exponent, 7))
    if key == ("Calculus", "Limits and Continuity"):
        return f"Find the limit of (x^2 - {i * i})/(x - {i}) as x approaches {i}.", "integer", _answer(2 * i)
    if key == ("Calculus", "Differentiation"):
        return f"Differentiate {i}*x^3 + {i + 1}*x with respect to x.", "symbolic_expression", f"{3 * i}*x^2 + {i + 1}"
    if key == ("Calculus", "Integration"):
        upper = i + 1
        value = Fraction(i * upper * upper, 2) + upper
        return f"Evaluate the integral of {i}*x + 1 from x = 0 to x = {upper}.", "rational", _answer(value)
    if key == ("Linear Algebra", "Matrix Operations"):
        return f"Add matrices [[{i}, 1], [2, {i + 1}]] and [[1, 2], [3, 4]].", "matrix", f"[[{i + 1}, 3], [5, {i + 5}]]"
    if key == ("Linear Algebra", "Linear Systems"):
        x, y, z = i, i + 1, i + 2
        return f"Solve for (x, y, z): x+y+z={x+y+z}, x-y={x-y}, and z-y={z-y}.", "ordered_tuple", f"({x}, {y}, {z})"
    if key == ("Linear Algebra", "Vectors and Vector Spaces"):
        return f"Find the dot product of ({i}, 2, -1) and (3, {i + 1}, 4).", "integer", _answer(3 * i + 2 * (i + 1) - 4)
    if key == ("Composite Comprehensive", "Cross-Domain Multi-Step Problems"):
        price, discount = 20 * i, index % 5 + 5
        return f"An item costs {price}; apply a {discount}% discount, then add 10% tax. Find the exact final price.", "rational", _answer(Fraction(price * (100 - discount) * 110, 10_000))
    if key == ("Composite Comprehensive", "Proof and Mathematical Reasoning"):
        k = i
        result = k % 2 == 1
        return f"True or false: for every integer n, n*(n + {k}) is even.", "boolean", _answer(result)
    if key == ("Composite Comprehensive", "Constraint Synthesis"):
        r3, r5 = index % 3, (index * 2 + 1) % 5
        solution = next(n for n in range(15) if n % 3 == r3 and n % 5 == r5)
        return f"Find the least nonnegative n with n mod 3 = {r3} and n mod 5 = {r5}, then add {15 * index}.", "integer", _answer(solution + 15 * index)
    raise KeyError(key)


def build_large_development_payload() -> dict[str, Any]:
    questions = []
    for category, subcategories in DEFAULT_TAXONOMY.items():
        for subcategory in subcategories:
            for index in range(20):
                question, answer_type, answer = _item(category, subcategory, index)
                contract = normalize_generated_gold_contract(question, answer, answer_type)
                questions.append(
                    {
                        "question_id": f"dev540-{len(questions) + 1:04d}",
                        "category": category,
                        "sub_category": subcategory,
                        "difficulty": index % 5 + 1,
                        "question": question,
                        "answer_type": contract["answer_type"],
                        "canonical_answer": contract["canonical_answer"],
                        "display_answer": contract["display_answer"],
                        "tolerance": contract["tolerance"],
                        "source_dataset": "project_native_synthetic",
                        "construction_method": "deterministic_formula_v1",
                        "verification": {
                            "status": "independently_verified",
                            "method": "closed_form_recomputation",
                            "source_id": f"formula-{len(questions) + 1:04d}",
                        },
                    }
                )
    return {
        "schema_version": "1.0",
        "name": "fixed_math_development_static_540_v1",
        "role": "development_regression",
        "description": "Deterministic stress/regression set; prohibited as a blind or paper test set.",
        "design_references": [
            {"name": "GSM8K", "questions_copied": False},
            {"name": "MATH", "questions_copied": False},
            {"name": "DeepMind Mathematics Dataset", "questions_copied": False},
        ],
        "questions": questions,
    }


def write_large_development_benchmark(output_path: str | Path, allowed_data_root: str | Path) -> dict[str, Any]:
    output = Path(output_path).expanduser().resolve()
    allowed = Path(allowed_data_root).expanduser().resolve()
    if not output.is_relative_to(allowed):
        raise ValueError("Development benchmark must stay under allowed_data_root")
    if output.exists():
        raise FileExistsError(f"Immutable benchmark already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(build_large_development_payload(), output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    manifest = {
        "schema_version": "1.0",
        "name": "fixed_math_development_static_540_v1",
        "benchmark_role": "development_regression",
        "question_count": 540,
        "subcategory_count": 27,
        "sha256": digest,
        "formal_evidence_prohibited": True,
    }
    atomic_json(manifest, output.with_suffix(".manifest.json"))
    if os.name != "nt":
        os.chmod(output, 0o600)
    return manifest
