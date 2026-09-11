"""Shared X/Z measurement-basis validation."""

from __future__ import annotations

from typing import Literal, TypeAlias, cast

LogicalBasis: TypeAlias = Literal["X", "Z"]


def normalize_measurement_basis(basis: str) -> LogicalBasis:
    """Return uppercase X or Z; reject any other measurement basis."""
    normalized = basis.upper()
    if normalized not in {"X", "Z"}:
        raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")
    return cast(LogicalBasis, normalized)
