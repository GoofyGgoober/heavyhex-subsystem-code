"""Argument validation shared across the package."""

from __future__ import annotations

from collections.abc import Sequence
from math import isfinite


def validate_probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}")


def validate_nonnegative_number(name: str, value: float) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number, got {value!r}")


def validate_binary_bits(name: str, bits: Sequence[int], width: int) -> None:
    if len(bits) != width or any(not isinstance(bit, int) or bit not in (0, 1) for bit in bits):
        raise ValueError(f"{name} must be {width} binary bits, got {tuple(bits)!r}")
