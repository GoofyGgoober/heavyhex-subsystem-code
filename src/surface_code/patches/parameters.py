"""Parameter calculations for the default rotated patch."""

from ..core import Pauli
from ..core.stabilizer import _independent_rows as _code_independent_rows
from ..core.stabilizer import _in_rowspace as _code_in_rowspace
from .rotated_d3 import PATCH

N = PATCH.code.n
_SYMPLECTIC_WIDTH = 2 * N


def _to_symplectic(pauli: Pauli) -> int:
    return PATCH.code.to_symplectic(pauli)


def _from_symplectic(bits: int) -> Pauli:
    return PATCH.code.from_symplectic(bits)


def _independent_rows(rows: list[int]) -> list[int]:
    return _code_independent_rows(rows, _SYMPLECTIC_WIDTH)


def _in_rowspace(bits: int, basis: list[int]) -> bool:
    return _code_in_rowspace(bits, tuple(basis))


_STABILIZER_BASIS = list(PATCH.code.stabilizer_basis)


def stabilizer_rank() -> int:
    return PATCH.code.stabilizer_rank()


def in_stabilizer_group(pauli: Pauli) -> bool:
    return PATCH.code.in_stabilizer_group(pauli)


def is_logical(pauli: Pauli) -> bool:
    """True for a non-trivial logical: commutes with every stabilizer, not a stabilizer."""
    return PATCH.code.is_logical(pauli)


def distance() -> int:
    return PATCH.code.distance()


def parameters() -> tuple[int, int, int]:
    return PATCH.code.parameters()
