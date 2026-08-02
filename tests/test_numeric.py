import math

from autobencher.numeric import finite_float, finite_int


def test_finite_float_rejects_untrusted_non_numeric_values():
    for value in (None, True, "", "not-a-number", math.nan, math.inf):
        assert finite_float(value) is None
    assert finite_float("0.75") == 0.75


def test_finite_int_does_not_silently_truncate():
    assert finite_int("3") == 3
    assert finite_int(3.0) == 3
    assert finite_int(3.5) is None
    assert finite_int(None, default=5) == 5
