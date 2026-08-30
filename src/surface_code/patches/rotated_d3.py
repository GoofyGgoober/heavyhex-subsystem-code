"""Definition of the [[9,1,3]] rotated planar surface-code patch."""

from ..core import Pauli
from .base import Check, SurfaceCodePatch

_DISTANCE = 3


def _qubit(row: int, col: int) -> int:
    return row * _DISTANCE + col


# X checks first and Z checks second preserves the public syndrome ordering.
_CHECKS = (
    Check("X", frozenset({_qubit(0, 1), _qubit(0, 2), _qubit(1, 1), _qubit(1, 2)}), 9),
    Check("X", frozenset({_qubit(1, 0), _qubit(1, 1), _qubit(2, 0), _qubit(2, 1)}), 10),
    Check("X", frozenset({_qubit(0, 0), _qubit(0, 1)}), 11),
    Check("X", frozenset({_qubit(2, 1), _qubit(2, 2)}), 12),
    Check("Z", frozenset({_qubit(0, 0), _qubit(0, 1), _qubit(1, 0), _qubit(1, 1)}), 13),
    Check("Z", frozenset({_qubit(1, 1), _qubit(1, 2), _qubit(2, 1), _qubit(2, 2)}), 14),
    Check("Z", frozenset({_qubit(1, 0), _qubit(2, 0)}), 15),
    Check("Z", frozenset({_qubit(0, 2), _qubit(1, 2)}), 16),
)

PATCH = SurfaceCodePatch(
    distance=_DISTANCE,
    data_qubits=tuple(range(_DISTANCE**2)),
    checks=_CHECKS,
    logical_x=Pauli.x_on(_qubit(row, 0) for row in range(_DISTANCE)),
    logical_z=Pauli.z_on(_qubit(0, col) for col in range(_DISTANCE)),
)
