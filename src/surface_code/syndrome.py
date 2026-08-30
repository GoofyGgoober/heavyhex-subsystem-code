"""One local syndrome round on the ideal 17-qubit patch.

Z-tiles: CNOT from each data qubit onto the ancilla, then measure Z.
X-tiles: H, CNOT from the ancilla onto each data qubit, H, then measure Z.

No IBM layout. Ancillas are fully connected to their tiles.
"""

from __future__ import annotations

from dataclasses import dataclass

from .ancillas import ANCILLAS, X_ANCILLAS, Z_ANCILLAS
from .checks import STABILIZERS, X_CHECKS, Z_CHECKS
from .pauli import Pauli


@dataclass(frozen=True)
class H:
    qubit: int


@dataclass(frozen=True)
class CX:
    control: int
    target: int


@dataclass(frozen=True)
class MeasureZ:
    qubit: int


def _syndrome_circuit() -> tuple[H | CX | MeasureZ, ...]:
    ops: list[H | CX | MeasureZ] = []
    for ancilla, support in zip(Z_ANCILLAS, Z_CHECKS):
        for data in sorted(support):
            ops.append(CX(data, ancilla))
    for ancilla, support in zip(X_ANCILLAS, X_CHECKS):
        ops.append(H(ancilla))
        for data in sorted(support):
            ops.append(CX(ancilla, data))
        ops.append(H(ancilla))
    for ancilla in ANCILLAS:
        ops.append(MeasureZ(ancilla))
    return tuple(ops)


SYNDROME_CIRCUIT = _syndrome_circuit()


def _conjugate_h(pauli: Pauli, qubit: int) -> Pauli:
    in_x = qubit in pauli.x
    in_z = qubit in pauli.z
    x = set(pauli.x)
    z = set(pauli.z)
    x.discard(qubit)
    z.discard(qubit)
    if in_z:
        x.add(qubit)
    if in_x:
        z.add(qubit)
    return Pauli(frozenset(x), frozenset(z))


def _conjugate_cx(pauli: Pauli, control: int, target: int) -> Pauli:
    x = set(pauli.x)
    z = set(pauli.z)
    if control in x:
        x.symmetric_difference_update((target,))
    if target in z:
        z.symmetric_difference_update((control,))
    return Pauli(frozenset(x), frozenset(z))


def abstract_syndrome(error: Pauli) -> tuple[int, ...]:
    """Syndrome from commutation: 1 iff the error anticommutes with that stabilizer."""
    return tuple(0 if error.commutes(stab) else 1 for stab in STABILIZERS)


def extract_syndrome(error: Pauli) -> tuple[int, ...]:
    """Push a data error through the circuit and read the 8 ancilla bits."""
    pauli = error
    bits: list[int] = []
    for op in SYNDROME_CIRCUIT:
        if isinstance(op, H):
            pauli = _conjugate_h(pauli, op.qubit)
        elif isinstance(op, CX):
            pauli = _conjugate_cx(pauli, op.control, op.target)
        else:
            bits.append(int(op.qubit in pauli.x))
    return tuple(bits)
