"""One local syndrome round on the ideal 17-qubit patch.

Z-tiles: CNOT from each data qubit onto the ancilla, then measure Z.
X-tiles: H, CNOT from the ancilla onto each data qubit, H, then measure Z.

No IBM layout. Ancillas are fully connected to their tiles, so the checks run
one after another and a round's duration is a model parameter, not a schedule.

CNOT order still matters: an ancilla fault halfway through a weight-4 check
leaves a two-qubit "hook" error on the data qubits it has not visited yet.
Z-tiles visit their qubits column by column and X-tiles row by row, so every
hook lies across the logical operator of its own type rather than along it,
and single faults cannot shorten the effective distance.
"""

from __future__ import annotations

from ..core import Pauli
from ..patches import PATCH
from .operations import CX, H, MeasureZ


def _column_major(qubit: int) -> tuple[int, int]:
    row, column = divmod(qubit, PATCH.distance)
    return column, row


def _syndrome_circuit() -> tuple[H | CX | MeasureZ, ...]:
    ops: list[H | CX | MeasureZ] = []
    for ancilla, support in zip(PATCH.z_ancillas, PATCH.z_checks):
        for data in sorted(support, key=_column_major):
            ops.append(CX(data, ancilla))
    for ancilla, support in zip(PATCH.x_ancillas, PATCH.x_checks):
        ops.append(H(ancilla))
        for data in sorted(support):
            ops.append(CX(ancilla, data))
        ops.append(H(ancilla))
    for ancilla in PATCH.ancillas:
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
    return PATCH.code.syndrome(error)


def extract_syndrome(error: Pauli) -> tuple[int, ...]:
    """Push a data error through the circuit and read the 8 ancilla bits."""
    PATCH.code.validate_data_pauli(error, name="error")
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
