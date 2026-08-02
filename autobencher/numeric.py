"""Fail-closed numeric coercion for untrusted model and artifact fields."""

from __future__ import annotations

import math
from typing import Any


def finite_float(value: Any, default: float | None = None) -> float | None:
    """Return a finite float, or ``default`` for missing/malformed values."""

    if isinstance(value, bool) or value is None:
        return default
    try:
        numeric = float(value)
    except (TypeError, ValueError, OverflowError):
        return default
    return numeric if math.isfinite(numeric) else default


def finite_int(value: Any, default: int | None = None) -> int | None:
    """Return an exact finite integer, or ``default`` without truncation."""

    numeric = finite_float(value)
    if numeric is None or not numeric.is_integer():
        return default
    try:
        return int(numeric)
    except (TypeError, ValueError, OverflowError):
        return default
