"""Parameter calculations for the default rotated patch."""

from ..core import Pauli
from .rotated_d3 import PATCH


def in_stabilizer_group(pauli: Pauli) -> bool:
    return PATCH.code.in_stabilizer_group(pauli)


def is_logical(pauli: Pauli) -> bool:
    """True for a non-trivial logical: commutes with every stabilizer, not a stabilizer."""
    return PATCH.code.is_logical(pauli)


def parameters() -> tuple[int, int, int]:
    return PATCH.code.parameters()
