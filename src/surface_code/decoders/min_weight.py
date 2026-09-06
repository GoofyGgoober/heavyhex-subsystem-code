"""Minimum-weight decoder: every syndrome maps to a cheapest data Pauli."""

from collections.abc import Iterator
from functools import cache
from itertools import combinations, product

from .._validation import validate_binary_bits
from ..core import Pauli, StabilizerCode
from ..patches import PATCH

Syndrome = tuple[int, ...]

_PAULI_COMPONENTS = ((True, False), (True, True), (False, True))  # X, Y, Z


def _errors_of_weight(weight: int, data_qubits: tuple[int, ...]) -> Iterator[Pauli]:
    """Yield physical Paulis in deterministic X/Y/Z order without storing them all."""
    for qubits in combinations(data_qubits, weight):
        for axes in product(_PAULI_COMPONENTS, repeat=weight):
            x: set[int] = set()
            z: set[int] = set()
            for qubit, (x_bit, z_bit) in zip(qubits, axes):
                if x_bit:
                    x.add(qubit)
                if z_bit:
                    z.add(qubit)
            yield Pauli(frozenset(x), frozenset(z))


@cache
def _min_weight_table(code: StabilizerCode) -> dict[Syndrome, Pauli]:
    # Redundant stabilizer generators impose constraints on the syndrome bits.
    size = 1 << code.stabilizer_rank()
    table: dict[Syndrome, Pauli] = {}
    for weight in range(code.n + 1):
        for error in _errors_of_weight(weight, code.data_qubits):
            table.setdefault(code.syndrome(error), error)
            if len(table) == size:
                return table
    raise RuntimeError(f"filled {len(table)} of {size} syndromes")


def decode(syndrome: Syndrome, code: StabilizerCode | None = None) -> Pauli:
    code = PATCH.code if code is None else code
    validate_binary_bits("syndrome", syndrome, code.syndrome_size)
    try:
        return _min_weight_table(code)[syndrome]
    except KeyError:
        raise ValueError(f"syndrome is inconsistent with the stabilizers: {syndrome!r}") from None


def z_basis_success(error: Pauli, code: StabilizerCode | None = None) -> int:
    """1 if leftover commutes with Z_L, 0 if it is an X_L."""
    code = PATCH.code if code is None else code
    leftover = decode(code.syndrome(error), code) * error
    return int(leftover.commutes(code.logical_z))
