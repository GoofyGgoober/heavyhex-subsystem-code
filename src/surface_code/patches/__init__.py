"""Concrete surface-code patch definitions and their shared models."""

from .base import Check, CheckBasis, SurfaceCodePatch
from .rotated_d3 import PATCH
from .parameters import is_logical, parameters

DISTANCE = PATCH.distance
DATA_QUBITS = PATCH.data_qubits
X_CHECKS = PATCH.x_checks
Z_CHECKS = PATCH.z_checks
X_ANCILLAS = PATCH.x_ancillas
Z_ANCILLAS = PATCH.z_ancillas
ANCILLAS = PATCH.ancillas
X_STABILIZERS = PATCH.x_stabilizers
Z_STABILIZERS = PATCH.z_stabilizers
STABILIZERS = PATCH.stabilizers
LOGICAL_X = PATCH.logical_x
LOGICAL_Z = PATCH.logical_z


def data_qubit(row: int, col: int) -> int:
    return PATCH.data_qubit(row, col)


__all__ = [
    "ANCILLAS",
    "DATA_QUBITS",
    "DISTANCE",
    "LOGICAL_X",
    "LOGICAL_Z",
    "PATCH",
    "STABILIZERS",
    "X_ANCILLAS",
    "X_CHECKS",
    "X_STABILIZERS",
    "Z_ANCILLAS",
    "Z_CHECKS",
    "Z_STABILIZERS",
    "Check",
    "CheckBasis",
    "SurfaceCodePatch",
    "data_qubit",
    "is_logical",
    "parameters",
]
