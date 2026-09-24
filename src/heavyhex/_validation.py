"""Argument checks used across the package."""

from __future__ import annotations

from collections.abc import Sequence

MAX_SEED = (1 << 63) - 1


def validate_probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}")


def validate_binary_bits(name: str, bits: Sequence[int], width: int) -> None:
    if len(bits) != width or any(not isinstance(bit, int) or bit not in (0, 1) for bit in bits):
        raise ValueError(f"{name} must be {width} binary bits, got {tuple(bits)!r}")


def validate_shots_and_seed(shots: int, seed: int | None) -> None:
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots!r}")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed > MAX_SEED
    ):
        raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}, or None, got {seed!r}")
