"""Minimum-weight decoder: every syndrome maps to a cheapest data Pauli."""

from functools import cache
from itertools import combinations, product

from ..core import Pauli, StabilizerCode
from ..patches import PATCH

Syndrome = tuple[int, ...]

_AXES = (
    (frozenset({0}), frozenset()),
    (frozenset({0}), frozenset({0})),
    (frozenset(), frozenset({0})),
)


def _errors_on_n_qubits(n: int, data_qubits: tuple[int, ...]) -> tuple[Pauli, ...]:
    if n == 0:
        return (Pauli(),)
    errors: list[Pauli] = []
    for qubits in combinations(data_qubits, n):
        for axes in product(_AXES, repeat=n):
            x: set[int] = set()
            z: set[int] = set()
            for qubit, (x_bit, z_bit) in zip(qubits, axes):
                if x_bit:
                    x.add(qubit)
                if z_bit:
                    z.add(qubit)
            errors.append(Pauli(frozenset(x), frozenset(z)))
    return tuple(errors)


@cache
def _min_weight_table(code: StabilizerCode) -> dict[Syndrome, Pauli]:
    size = 1 << code.syndrome_size
    table: dict[Syndrome, Pauli] = {}
    for weight in range(code.n + 1):
        for error in _errors_on_n_qubits(weight, code.data_qubits):
            table.setdefault(code.syndrome(error), error)
        if len(table) == size:
            return table
    raise RuntimeError(f"filled {len(table)} of {size} syndromes")


def decode(syndrome: Syndrome, code: StabilizerCode | None = None) -> Pauli:
    code = PATCH.code if code is None else code
    if len(syndrome) != code.syndrome_size or any(bit not in (0, 1) for bit in syndrome):
        raise ValueError(f"syndrome must be {code.syndrome_size} bits, got {syndrome!r}")
    return _min_weight_table(code)[syndrome]


def z_basis_success(error: Pauli, code: StabilizerCode | None = None) -> int:
    """1 if leftover commutes with Z_L, 0 if it is an X_L."""
    code = PATCH.code if code is None else code
    leftover = decode(code.syndrome(error), code) * error
    return int(leftover.commutes(code.logical_z))
