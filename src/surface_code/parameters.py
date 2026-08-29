"""Identify the patch as [[n, k, d]]."""

from .checks import STABILIZERS
from .pauli import Pauli
from .qubits import DATA_QUBITS

N = len(DATA_QUBITS)
_SYMPLECTIC_WIDTH = 2 * N


def _to_symplectic(pauli: Pauli) -> int:
    bits = 0
    for q in pauli.x:
        bits |= 1 << q
    for q in pauli.z:
        bits |= 1 << (N + q)
    return bits


def _from_symplectic(bits: int) -> Pauli:
    return Pauli(
        x=frozenset(q for q in range(N) if bits & (1 << q)),
        z=frozenset(q for q in range(N) if bits & (1 << (N + q))),
    )


def _independent_rows(rows: list[int]) -> list[int]:
    rows = list(rows)
    lead = 0
    row = 0
    while row < len(rows) and lead < _SYMPLECTIC_WIDTH:
        pivot = next((i for i in range(row, len(rows)) if rows[i] & (1 << lead)), None)
        if pivot is None:
            lead += 1
            continue
        rows[row], rows[pivot] = rows[pivot], rows[row]
        for i, other in enumerate(rows):
            if i != row and other & (1 << lead):
                rows[i] ^= rows[row]
        row += 1
        lead += 1
    return [r for r in rows if r]


def _in_rowspace(bits: int, basis: list[int]) -> bool:
    leftover = bits
    for row in basis:
        pivot = row & -row
        if leftover & pivot:
            leftover ^= row
    return leftover == 0


_STABILIZER_BASIS = _independent_rows([_to_symplectic(s) for s in STABILIZERS])


def stabilizer_rank() -> int:
    return len(_STABILIZER_BASIS)


def in_stabilizer_group(pauli: Pauli) -> bool:
    return _in_rowspace(_to_symplectic(pauli), _STABILIZER_BASIS)


def is_logical(pauli: Pauli) -> bool:
    """True for a non-trivial logical: commutes with every stabilizer, not a stabilizer."""
    return all(pauli.commutes(s) for s in STABILIZERS) and not in_stabilizer_group(pauli)


def distance() -> int:
    best = N
    for bits in range(1, 1 << _SYMPLECTIC_WIDTH):
        pauli = _from_symplectic(bits)
        w = pauli.weight()
        if w >= best:
            continue
        if is_logical(pauli):
            best = w
    return best


def parameters() -> tuple[int, int, int]:
    k = N - stabilizer_rank()
    return (N, k, distance())
