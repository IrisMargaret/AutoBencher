"""Deterministic mathematical truth generation for the Flywheel pipeline."""

from __future__ import annotations

import contextlib
import hashlib
import os
import re
import signal
import threading
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


class FailureType(str, Enum):
    """Stable structured failure labels used by generation monitoring."""

    TRUTH_PARSE_FAIL = "truth_parse_fail"
    NO_CLOSED_SOLUTION = "no_closed_solution"
    INFINITE_SOLUTIONS = "infinite_solutions"
    SOLVE_TIMEOUT = "solve_timeout"
    PARTIAL_SOLUTION = "partial_solution"
    REPAIR_EXHAUSTED = "repair_exhausted"
    GENERATOR_FORMAT_ERROR = "generator_format_error"
    SUBCATEGORY_COOLDOWN = "subcategory_cooldown"


class TruthSolverError(RuntimeError):
    """Base class for deterministic truth-generation failures."""


class TruthParseError(TruthSolverError):
    pass


class NoClosedFormError(TruthSolverError):
    pass


class InfiniteSolutionsError(TruthSolverError):
    pass


class SolveTimeoutError(TruthSolverError):
    pass


@dataclass
class TruthSolveResult:
    success: bool
    canonical_answer: str | None
    answer_type: str | None
    truth_validation_details: dict[str, Any] = field(default_factory=dict)
    failure_type: str | None = None
    failure_summary: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ExtractedMath:
    route: str
    original_question: str
    normalized_expressions: list[str]
    variable_names: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)


class MathExpressionPreprocessor:
    """Extract solver-safe expressions while preserving the source question."""

    _ALLOWED_EXPRESSION = re.compile(r"^[A-Za-z0-9_+\-*/^().\s]+$")
    _VARIABLE_TUPLE = re.compile(
        r"(?:for|variables?)\s*\(([^()]*)\)",
        flags=re.IGNORECASE,
    )
    _INTEGRAL = re.compile(
        r"(?:integrate|evaluate\s+the\s+integral)\s+"
        r"(?P<expression>.+?)\s+with\s+respect\s+to\s+"
        r"(?P<variable>[A-Za-z]\w*)"
        r"(?:\s+from\s+(?P<lower>.+?)\s+to\s+(?P<upper>.+?))?[.?\s]*$",
        flags=re.IGNORECASE,
    )
    _LIMIT = re.compile(
        r"(?:find|evaluate|compute)?\s*(?:the\s+)?limit\s+(?:of\s+)?"
        r"(?P<expression>.+?)\s+as\s+(?P<variable>[A-Za-z]\w*)\s+"
        r"(?:approaches|tends\s+to)\s+(?P<point>.+?)[.?\s]*$",
        flags=re.IGNORECASE,
    )
    _DIRECT = re.compile(
        r"(?:what\s+is|evaluate|compute|calculate|simplify)\s*:?\s*"
        r"(?P<expression>.+?)[.?\s]*$",
        flags=re.IGNORECASE,
    )
    _KNOWN_NAMES = {
        "sin",
        "cos",
        "tan",
        "sqrt",
        "exp",
        "log",
        "pi",
        "e",
    }

    @staticmethod
    def normalize_syntax(expression: str) -> str:
        expression = str(expression).strip()
        expression = expression.replace("−", "-").replace("×", "*")
        expression = expression.replace("÷", "/").replace("^", "**")
        expression = re.sub(r"\s+", " ", expression)
        return expression.strip().rstrip(".?")

    @staticmethod
    def split_top_level(text: str) -> list[str]:
        parts: list[str] = []
        start = 0
        depth = 0
        for index, character in enumerate(text):
            if character in "([{":
                depth += 1
            elif character in ")]}":
                depth = max(0, depth - 1)
            elif depth == 0 and character in ",;\n":
                part = text[start:index].strip()
                if part:
                    parts.append(part)
                start = index + 1
        final = text[start:].strip()
        if final:
            parts.append(final)
        return parts

    def extract(self, question: str) -> ExtractedMath:
        original = str(question)
        if not original.strip():
            raise TruthParseError("The question is empty.")
        if len(original) > 16000:
            raise TruthParseError("The question exceeds the parser limit.")

        limit_match = self._LIMIT.search(original)
        if limit_match:
            expression = self.normalize_syntax(
                limit_match.group("expression")
            )
            point = self.normalize_syntax(limit_match.group("point"))
            self._assert_safe(expression)
            self._assert_safe(point)
            variable = limit_match.group("variable")
            return ExtractedMath(
                route="limit",
                original_question=original,
                normalized_expressions=[expression],
                variable_names=[variable],
                metadata={"point": point},
            )

        integral_match = self._INTEGRAL.search(original)
        if integral_match:
            expression = self.normalize_syntax(
                integral_match.group("expression")
            )
            self._assert_safe(expression)
            metadata: dict[str, Any] = {}
            if integral_match.group("lower") is not None:
                lower = self.normalize_syntax(
                    integral_match.group("lower")
                )
                upper = self.normalize_syntax(
                    integral_match.group("upper")
                )
                self._assert_safe(lower)
                self._assert_safe(upper)
                metadata.update({"lower": lower, "upper": upper})
            return ExtractedMath(
                route="integral",
                original_question=original,
                normalized_expressions=[expression],
                variable_names=[integral_match.group("variable")],
                metadata=metadata,
            )

        equations = self._extract_equations(original)
        if equations:
            variables = self._extract_variables(original, equations)
            return ExtractedMath(
                route=(
                    "single_equation"
                    if len(equations) == 1
                    else "equation_system"
                ),
                original_question=original,
                normalized_expressions=equations,
                variable_names=variables,
            )

        direct_match = self._DIRECT.search(original)
        if direct_match:
            expression = self.normalize_syntax(
                direct_match.group("expression")
            )
            self._assert_safe(expression)
            variables = sorted(
                self._identifiers(expression) - self._KNOWN_NAMES
            )
            return ExtractedMath(
                route="direct_expression",
                original_question=original,
                normalized_expressions=[expression],
                variable_names=variables,
            )
        raise TruthParseError(
            "No supported equation, system, integral, limit, or direct "
            "expression was extracted."
        )

    def _extract_equations(self, question: str) -> list[str]:
        candidate = question
        if ":" in question:
            tail = question.rsplit(":", 1)[1]
            if "=" in tail:
                candidate = tail
        candidate = re.sub(
            r"\s+\band\b\s+(?=[A-Za-z]\w*\s*[+\-*/^=])",
            ", ",
            candidate,
            flags=re.IGNORECASE,
        )
        equations = []
        for part in self.split_top_level(candidate):
            if "=" not in part:
                continue
            normalized = re.sub(
                r"^\s*(?:eq(?:uation)?\s*\d+|[\[(]?\d+[\])]?)[.:]\s*",
                "",
                part,
                flags=re.IGNORECASE,
            )
            normalized = self.normalize_syntax(normalized)
            if normalized.count("=") != 1:
                raise TruthParseError(
                    f"Expected one equality operator: {part}"
                )
            left, right = (
                item.strip() for item in normalized.split("=", 1)
            )
            self._assert_safe(left)
            self._assert_safe(right)
            equations.append(f"{left} = {right}")
        if len(equations) > 20:
            raise TruthParseError("The equation count exceeds the limit.")
        return equations

    def _extract_variables(
        self,
        question: str,
        equations: list[str],
    ) -> list[str]:
        match = self._VARIABLE_TUPLE.search(question)
        if match:
            names = re.findall(r"[A-Za-z]\w*", match.group(1))
        else:
            names = sorted(
                set().union(
                    *(self._identifiers(item) for item in equations)
                )
                - self._KNOWN_NAMES
            )
        names = list(dict.fromkeys(names))
        if not names or len(names) > 10:
            raise TruthParseError(
                "Unable to determine a bounded variable list."
            )
        return names

    @staticmethod
    def _identifiers(expression: str) -> set[str]:
        return set(re.findall(r"[A-Za-z]\w*", expression))

    def _assert_safe(self, expression: str) -> None:
        if (
            not expression
            or "__" in expression
            or not self._ALLOWED_EXPRESSION.fullmatch(expression)
        ):
            raise TruthParseError(
                f"Unsupported expression syntax: {expression}"
            )


class TruthSolver:
    """Authoritative SymPy-backed source of canonical mathematical answers."""

    def __init__(
        self,
        solve_timeout: float = 10.0,
        absolute_tolerance: float = 1.0e-6,
        max_retry: int = 1,
    ):
        if solve_timeout <= 0:
            raise ValueError("solve_timeout must be positive")
        if max_retry < 1:
            raise ValueError("max_retry must be positive")
        self.solve_timeout = float(solve_timeout)
        self.absolute_tolerance = float(absolute_tolerance)
        self.max_retry = int(max_retry)
        self.preprocessor = MathExpressionPreprocessor()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TruthSolver":
        generation = config["generation"]
        return cls(
            solve_timeout=float(
                generation["truth_solver_timeout_seconds"]
            ),
            absolute_tolerance=float(
                config["answer_normalization"]["absolute_tolerance"]
            ),
            max_retry=int(generation["truth_solver_max_retry"]),
        )

    def solve(self, question: str) -> TruthSolveResult:
        source = str(question)
        source_hash = hashlib.sha256(
            source.encode("utf-8")
        ).hexdigest()
        started = time.monotonic()
        try:
            with self._timeout():
                extracted = self.preprocessor.extract(source)
                result = self._dispatch(extracted)
            elapsed = time.monotonic() - started
            result.truth_validation_details.update(
                {
                    "source_question": source,
                    "source_question_sha256": source_hash,
                    "solver_question_sha256": source_hash,
                    "route": extracted.route,
                    "normalized_expressions": (
                        extracted.normalized_expressions
                    ),
                    "elapsed_seconds": round(elapsed, 6),
                }
            )
            if elapsed > self.solve_timeout:
                raise SolveTimeoutError(
                    "Truth solving exceeded the configured timeout."
                )
            return result
        except SolveTimeoutError as exc:
            return self._failure(
                FailureType.SOLVE_TIMEOUT,
                exc,
                source,
                source_hash,
                started,
            )
        except InfiniteSolutionsError as exc:
            return self._failure(
                FailureType.INFINITE_SOLUTIONS,
                exc,
                source,
                source_hash,
                started,
            )
        except NoClosedFormError as exc:
            return self._failure(
                FailureType.NO_CLOSED_SOLUTION,
                exc,
                source,
                source_hash,
                started,
            )
        except (TruthParseError, ValueError, TypeError, SyntaxError) as exc:
            return self._failure(
                FailureType.TRUTH_PARSE_FAIL,
                exc,
                source,
                source_hash,
                started,
            )
        except Exception as exc:
            return self._failure(
                FailureType.NO_CLOSED_SOLUTION,
                exc,
                source,
                source_hash,
                started,
            )

    def validate_candidate(
        self,
        question: str,
        candidate_answer: Any,
    ) -> dict[str, Any]:
        """Validate a candidate and detect partial equation-system solutions."""
        truth = self.solve(question)
        result = {
            "verification_passed": False,
            "substitution_passed": False,
            "failure_type": truth.failure_type,
            "canonical_answer": truth.canonical_answer,
            "truth_validation_details": truth.truth_validation_details,
            "equations_passed": 0,
            "equations_total": 0,
        }
        if not truth.success:
            return result
        extracted = self.preprocessor.extract(question)
        if extracted.route != "equation_system":
            result["verification_passed"] = (
                str(candidate_answer).strip()
                == str(truth.canonical_answer).strip()
            )
            result["substitution_passed"] = result[
                "verification_passed"
            ]
            result["failure_type"] = (
                None
                if result["verification_passed"]
                else FailureType.NO_CLOSED_SOLUTION.value
            )
            return result
        try:
            parsed = self._parse_system(extracted)
            candidate = self._parse_tuple(candidate_answer, parsed)
            details = self._substitute_system(parsed, candidate)
        except Exception as exc:
            result.update(
                {
                    "failure_type": FailureType.TRUTH_PARSE_FAIL.value,
                    "candidate_parse_error": (
                        f"{type(exc).__name__}: {exc}"
                    ),
                }
            )
            return result
        passed_count = sum(item["passed"] for item in details)
        total_count = len(details)
        partial = 0 < passed_count < total_count
        result.update(
            {
                "verification_passed": (
                    passed_count == total_count
                    and self._tuple_text(candidate)
                    == truth.canonical_answer
                ),
                "substitution_passed": passed_count == total_count,
                "failure_type": (
                    FailureType.PARTIAL_SOLUTION.value
                    if partial
                    else (
                        None
                        if passed_count == total_count
                        else FailureType.NO_CLOSED_SOLUTION.value
                    )
                ),
                "equations_passed": passed_count,
                "equations_total": total_count,
                "substitution_details": details,
            }
        )
        return result

    def _dispatch(self, extracted: ExtractedMath) -> TruthSolveResult:
        if extracted.route == "equation_system":
            return self._solve_system(extracted)
        if extracted.route == "single_equation":
            return self._solve_single_equation(extracted)
        if extracted.route == "integral":
            return self._solve_integral(extracted)
        if extracted.route == "limit":
            return self._solve_limit(extracted)
        if extracted.route == "direct_expression":
            return self._solve_direct_expression(extracted)
        raise NoClosedFormError(f"Unsupported solver route: {extracted.route}")

    def _sympy(self):
        os.environ.setdefault("MPMATH_NOGMPY", "1")
        os.environ.setdefault("SYMPY_GROUND_TYPES", "python")
        import sympy
        from sympy.parsing.sympy_parser import (
            convert_xor,
            implicit_multiplication_application,
            parse_expr,
            standard_transformations,
        )

        transformations = standard_transformations + (
            implicit_multiplication_application,
            convert_xor,
        )
        return sympy, parse_expr, transformations

    def _parse_context(self, variable_names: list[str]):
        sympy, parse_expr, transformations = self._sympy()
        symbols = sympy.symbols(" ".join(variable_names))
        symbols = (symbols,) if len(variable_names) == 1 else tuple(symbols)
        local = dict(zip(variable_names, symbols))
        local.update(
            {
                "sin": sympy.sin,
                "cos": sympy.cos,
                "tan": sympy.tan,
                "sqrt": sympy.sqrt,
                "exp": sympy.exp,
                "log": sympy.log,
                "pi": sympy.pi,
                "e": sympy.E,
            }
        )
        return sympy, parse_expr, transformations, symbols, local

    def _parse_expression(
        self,
        text: str,
        parse_expr,
        transformations,
        local,
    ):
        unknown = (
            set(re.findall(r"[A-Za-z]\w*", text))
            - set(local)
        )
        if unknown:
            raise TruthParseError(
                "Unknown identifiers: " + ", ".join(sorted(unknown))
            )
        return parse_expr(
            text,
            local_dict=local,
            transformations=transformations,
            evaluate=True,
        )

    def _parse_system(self, extracted: ExtractedMath):
        (
            sympy,
            parse_expr,
            transformations,
            symbols,
            local,
        ) = self._parse_context(extracted.variable_names)
        equations = []
        for source in extracted.normalized_expressions:
            left_text, right_text = (
                item.strip() for item in source.split("=", 1)
            )
            left = self._parse_expression(
                left_text,
                parse_expr,
                transformations,
                local,
            )
            right = self._parse_expression(
                right_text,
                parse_expr,
                transformations,
                local,
            )
            equations.append(sympy.Eq(left, right, evaluate=False))
        return {
            "sympy": sympy,
            "symbols": symbols,
            "local": local,
            "equations": equations,
            "sources": extracted.normalized_expressions,
        }

    def _solve_system(self, extracted: ExtractedMath) -> TruthSolveResult:
        parsed = self._parse_system(extracted)
        sympy = parsed["sympy"]
        expressions = [
            sympy.simplify(item.lhs - item.rhs)
            for item in parsed["equations"]
        ]
        try:
            matrix_a, matrix_b = sympy.linear_eq_to_matrix(
                expressions,
                parsed["symbols"],
            )
            solution_set = sympy.linsolve(
                (matrix_a, matrix_b),
                parsed["symbols"],
            )
            route = "linear_system"
        except Exception:
            solution_set = sympy.nonlinsolve(
                expressions,
                parsed["symbols"],
            )
            route = "polynomial_system"
        solution = self._unique_tuple_solution(solution_set)
        verification = self._substitute_system(parsed, solution)
        if not verification or not all(item["passed"] for item in verification):
            raise NoClosedFormError(
                "The independently solved tuple failed truth substitution."
            )
        return TruthSolveResult(
            success=True,
            canonical_answer=self._tuple_text(solution),
            answer_type="ordered_tuple",
            truth_validation_details={
                "solver_branch": route,
                "substitution_passed": True,
                "substitution_details": verification,
            },
        )

    def _solve_single_equation(
        self,
        extracted: ExtractedMath,
    ) -> TruthSolveResult:
        parsed = self._parse_system(extracted)
        sympy = parsed["sympy"]
        variable = parsed["symbols"][0]
        expression = sympy.simplify(
            parsed["equations"][0].lhs - parsed["equations"][0].rhs
        )
        solution_set = sympy.solveset(
            expression,
            variable,
            domain=sympy.S.Complexes,
        )
        if solution_set is sympy.S.EmptySet:
            raise NoClosedFormError("The equation has no solution.")
        if not isinstance(solution_set, sympy.FiniteSet):
            raise InfiniteSolutionsError(
                "The equation does not have a finite closed solution set."
            )
        values = sorted(
            (sympy.simplify(item) for item in solution_set),
            key=sympy.default_sort_key,
        )
        if not values:
            raise NoClosedFormError("The equation has no solution.")
        if len(values) == 1:
            canonical = self._format_expr(values[0])
            answer_type = self._scalar_answer_type(values[0])
        else:
            canonical = "{" + ", ".join(
                self._format_expr(item) for item in values
            ) + "}"
            answer_type = "set"
        checks = []
        for value in values:
            left = sympy.simplify(
                parsed["equations"][0].lhs.subs(variable, value)
            )
            right = sympy.simplify(
                parsed["equations"][0].rhs.subs(variable, value)
            )
            checks.append(
                {
                    "original_equation": parsed["sources"][0],
                    "candidate": self._format_expr(value),
                    "substituted_left": self._format_expr(left),
                    "substituted_right": self._format_expr(right),
                    "difference": self._format_expr(
                        sympy.simplify(left - right)
                    ),
                    "passed": self._values_equal(left, right),
                }
            )
        if not all(item["passed"] for item in checks):
            raise NoClosedFormError(
                "The independent equation solution failed substitution."
            )
        return TruthSolveResult(
            success=True,
            canonical_answer=canonical,
            answer_type=answer_type,
            truth_validation_details={
                "solver_branch": "single_equation",
                "substitution_passed": True,
                "substitution_details": checks,
            },
        )

    def _solve_integral(self, extracted: ExtractedMath) -> TruthSolveResult:
        (
            sympy,
            parse_expr,
            transformations,
            symbols,
            local,
        ) = self._parse_context(extracted.variable_names)
        expression = self._parse_expression(
            extracted.normalized_expressions[0],
            parse_expr,
            transformations,
            local,
        )
        variable = symbols[0]
        if "lower" in extracted.metadata:
            lower = self._parse_expression(
                extracted.metadata["lower"],
                parse_expr,
                transformations,
                local,
            )
            upper = self._parse_expression(
                extracted.metadata["upper"],
                parse_expr,
                transformations,
                local,
            )
            answer = sympy.integrate(expression, (variable, lower, upper))
            branch = "definite_integral"
        else:
            answer = sympy.integrate(expression, variable)
            branch = "indefinite_integral"
        if answer.has(sympy.Integral):
            raise NoClosedFormError("SymPy returned an unevaluated integral.")
        return TruthSolveResult(
            success=True,
            canonical_answer=self._format_expr(answer),
            answer_type=(
                "symbolic_expression"
                if getattr(answer, "free_symbols", set())
                else self._scalar_answer_type(answer)
            ),
            truth_validation_details={
                "solver_branch": branch,
                "substitution_passed": True,
            },
        )

    def _solve_limit(self, extracted: ExtractedMath) -> TruthSolveResult:
        (
            sympy,
            parse_expr,
            transformations,
            symbols,
            local,
        ) = self._parse_context(extracted.variable_names)
        expression = self._parse_expression(
            extracted.normalized_expressions[0],
            parse_expr,
            transformations,
            local,
        )
        point = self._parse_expression(
            extracted.metadata["point"],
            parse_expr,
            transformations,
            local,
        )
        answer = sympy.limit(expression, symbols[0], point)
        if answer.has(sympy.Limit) or answer in {
            sympy.nan,
            sympy.zoo,
        }:
            raise NoClosedFormError("SymPy returned no closed finite limit.")
        return TruthSolveResult(
            success=True,
            canonical_answer=self._format_expr(answer),
            answer_type=self._scalar_answer_type(answer),
            truth_validation_details={
                "solver_branch": "limit",
                "substitution_passed": True,
            },
        )

    def _solve_direct_expression(
        self,
        extracted: ExtractedMath,
    ) -> TruthSolveResult:
        (
            sympy,
            parse_expr,
            transformations,
            _,
            local,
        ) = self._parse_context(extracted.variable_names or ["x"])
        expression = self._parse_expression(
            extracted.normalized_expressions[0],
            parse_expr,
            transformations,
            local,
        )
        answer = sympy.simplify(expression)
        if answer.free_symbols:
            answer_type = "symbolic_expression"
        else:
            answer_type = self._scalar_answer_type(answer)
        return TruthSolveResult(
            success=True,
            canonical_answer=self._format_expr(answer),
            answer_type=answer_type,
            truth_validation_details={
                "solver_branch": "direct_expression",
                "substitution_passed": True,
            },
        )

    def _unique_tuple_solution(self, solution_set):
        sympy, _, _ = self._sympy()
        if solution_set is sympy.S.EmptySet or solution_set == sympy.S.EmptySet:
            raise NoClosedFormError("The system has no solution.")
        solutions = list(solution_set)
        if len(solutions) != 1:
            raise NoClosedFormError(
                "The system does not have exactly one solution tuple."
            )
        solution = tuple(sympy.simplify(item) for item in solutions[0])
        if any(item.free_symbols for item in solution):
            raise InfiniteSolutionsError(
                "The system has infinitely many solutions."
            )
        return solution

    def _parse_tuple(self, candidate: Any, parsed) -> tuple[Any, ...]:
        sympy = parsed["sympy"]
        text = str(candidate).strip()
        text = text.replace(r"\left", "").replace(r"\right", "")
        text = text.strip("$").strip()
        if not (
            len(text) >= 2
            and text[0] in "(["
            and text[-1] in ")]"
        ):
            raise TruthParseError("The candidate is not an ordered tuple.")
        components = self.preprocessor.split_top_level(text[1:-1])
        if len(components) != len(parsed["symbols"]):
            raise TruthParseError(
                "The tuple length does not match the variable count."
            )
        values = []
        for component in components:
            component = re.sub(
                r"^[A-Za-z]\w*\s*=\s*",
                "",
                component.strip(),
            )
            self.preprocessor._assert_safe(component)
            value = sympy.sympify(
                component.replace("^", "**"),
                locals=parsed["local"],
            )
            if value.free_symbols:
                raise TruthParseError(
                    "Tuple components must be fully specified."
                )
            values.append(sympy.simplify(value))
        return tuple(values)

    def _substitute_system(self, parsed, candidate) -> list[dict[str, Any]]:
        sympy = parsed["sympy"]
        substitutions = dict(zip(parsed["symbols"], candidate))
        details = []
        for index, (source, equation) in enumerate(
            zip(parsed["sources"], parsed["equations"]),
            start=1,
        ):
            left = sympy.simplify(equation.lhs.subs(substitutions))
            right = sympy.simplify(equation.rhs.subs(substitutions))
            difference = sympy.simplify(left - right)
            details.append(
                {
                    "equation_index": index,
                    "original_equation": source,
                    "substituted_left": self._format_expr(left),
                    "substituted_right": self._format_expr(right),
                    "difference": self._format_expr(difference),
                    "passed": self._values_equal(left, right),
                }
            )
        return details

    def _values_equal(self, left, right) -> bool:
        sympy, _, _ = self._sympy()
        difference = sympy.simplify(left - right)
        if difference == 0 or difference.is_zero is True:
            return True
        if difference.free_symbols:
            return False
        try:
            return (
                abs(float(sympy.N(difference)))
                <= self.absolute_tolerance
            )
        except (TypeError, ValueError, OverflowError):
            return False

    def _format_expr(self, expression) -> str:
        sympy, _, _ = self._sympy()
        return sympy.sstr(sympy.simplify(expression))

    def _tuple_text(self, values) -> str:
        return "(" + ", ".join(self._format_expr(item) for item in values) + ")"

    def _scalar_answer_type(self, value) -> str:
        sympy, _, _ = self._sympy()
        value = sympy.simplify(value)
        if value.is_Integer:
            return "integer"
        if value.is_Rational:
            return "rational"
        if value.is_real is True and not value.free_symbols:
            return "decimal"
        return "symbolic_expression"

    def _failure(
        self,
        failure_type: FailureType,
        error: Exception,
        source: str,
        source_hash: str,
        started: float,
    ) -> TruthSolveResult:
        return TruthSolveResult(
            success=False,
            canonical_answer=None,
            answer_type=None,
            failure_type=failure_type.value,
            failure_summary=f"{type(error).__name__}: {error}",
            truth_validation_details={
                "source_question": source,
                "source_question_sha256": source_hash,
                "solver_question_sha256": source_hash,
                "elapsed_seconds": round(
                    time.monotonic() - started,
                    6,
                ),
            },
        )

    @contextlib.contextmanager
    def _timeout(self):
        can_interrupt = (
            hasattr(signal, "SIGALRM")
            and hasattr(signal, "setitimer")
            and threading.current_thread() is threading.main_thread()
        )
        if not can_interrupt:
            yield
            return

        def raise_timeout(signum, frame):
            del signum, frame
            raise SolveTimeoutError(
                "Truth solving exceeded the configured timeout."
            )

        previous = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, raise_timeout)
        signal.setitimer(signal.ITIMER_REAL, self.solve_timeout)
        try:
            yield
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous)
