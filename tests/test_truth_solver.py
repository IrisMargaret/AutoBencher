from autobencher.truth_solver import (
    FailureType,
    MathExpressionPreprocessor,
    TruthSolver,
)


def solver():
    return TruthSolver(solve_timeout=5, absolute_tolerance=1.0e-8)


def test_preprocessor_strips_language_and_normalizes_power():
    extracted = MathExpressionPreprocessor().extract(
        "Solve for x: x^2 - 5x + 6 = 0."
    )
    assert extracted.route == "single_equation"
    assert extracted.normalized_expressions == ["x**2 - 5x + 6 = 0"]


def test_linear_system_truth_is_independent_of_faulty_candidate():
    question = (
        "Solve the following system of equations for (x, y, z): "
        "x + 2y - z = 5, 2x - y + 3z = 4, "
        "-x + 3y + 2z = 7."
    )
    truth = solver().solve(question)
    assert truth.success is True
    assert truth.canonical_answer == "(8/5, 11/5, 1)"
    assert truth.answer_type == "ordered_tuple"
    candidate = solver().validate_candidate(question, "(2, 1, -1)")
    assert candidate["verification_passed"] is False
    assert candidate["substitution_passed"] is False
    assert candidate["failure_type"] == FailureType.PARTIAL_SOLUTION.value
    assert candidate["equations_passed"] == 1
    assert candidate["equations_total"] == 3
    assert [
        item["difference"]
        for item in candidate["substitution_details"]
    ] == ["0", "-4", "-8"]


def test_single_polynomial_equation_returns_finite_set():
    result = solver().solve("Solve for x: x^2 - 5x + 6 = 0.")
    assert result.success is True
    assert result.canonical_answer == "{2, 3}"
    assert result.answer_type == "set"


def test_integral_and_limit_use_separate_solver_routes():
    integral = solver().solve(
        "Integrate x^2 with respect to x from 0 to 3."
    )
    limit = solver().solve(
        "Find the limit of (x^2 - 1)/(x - 1) as x approaches 1."
    )
    assert integral.success is True
    assert integral.canonical_answer == "9"
    assert integral.truth_validation_details["route"] == "integral"
    assert limit.success is True
    assert limit.canonical_answer == "2"
    assert limit.truth_validation_details["route"] == "limit"


def test_derivative_inequality_and_exact_function_routes():
    derivative = solver().solve(
        "Differentiate x^3 + 2*x with respect to x at x = 2."
    )
    inequality = solver().solve(
        "Solve the inequality for x: x^2 - 5*x + 6 <= 0."
    )
    matrix = solver().solve(
        "Compute det(Matrix([[1,2],[3,4]]))."
    )
    statistics = solver().solve(
        "A sample contains 2, 4, and 6. Compute variance(2,4,6)."
    )
    assert derivative.canonical_answer == "14"
    assert derivative.truth_validation_details["route"] == "derivative"
    assert inequality.canonical_answer == "[2, 3]"
    assert inequality.answer_type == "interval"
    assert matrix.canonical_answer == "-2"
    assert statistics.canonical_answer == "8/3"


def test_exact_irrational_constant_is_symbolic_not_decimal():
    result = solver().solve("Compute sqrt(2).")
    assert result.success is True
    assert result.canonical_answer == "sqrt(2)"
    assert result.answer_type == "symbolic_expression"


def test_sympy_result_produces_training_safe_answer_anchored_reasoning():
    result = solver().solve(
        "Solve the system for (x, y): 2*x + y = 11, x - y = 1."
    )
    steps = solver().training_reasoning(result)
    assert len(steps) == 2
    assert "(4, 3)" in " ".join(steps)
    assert "residual is 0" in steps[1]


def test_infinite_system_and_unparseable_question_fail_closed():
    infinite = solver().solve(
        "Solve the system for (x, y): x + y = 2, 2x + 2y = 4."
    )
    unparseable = solver().solve(
        "A vague puzzle has an answer somewhere."
    )
    assert infinite.success is False
    assert infinite.failure_type == FailureType.INFINITE_SOLUTIONS.value
    assert unparseable.success is False
    assert unparseable.failure_type == FailureType.TRUTH_PARSE_FAIL.value
