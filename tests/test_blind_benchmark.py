from autobencher.blind_benchmark import _collision


def test_blind_dedup_rejects_parameterized_math_template():
    candidate = {
        "question": "Differentiate (x^2 + 3)*exp(2*x) with respect to x."
    }
    development = {
        "question": "Differentiate (x^2 + 1)*exp(x) with respect to x."
    }
    assert _collision(candidate, development) in {
        "parameterized_template",
        "math_structure",
    }


def test_blind_dedup_rejects_lexical_rewrite():
    candidate = {"question": "Compute the exact sum of 10 and 20."}
    development = {"question": "Compute the exact sum of 10 and 20."}
    assert _collision(candidate, development) == "exact_text"
