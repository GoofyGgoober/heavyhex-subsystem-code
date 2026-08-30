"""Generic stabilizer-code algebra, independent of patch geometry and circuits."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from .pauli import Pauli


def _independent_rows(rows: list[int], width: int) -> list[int]:
    rows = list(rows)
    lead = 0
    row = 0
    while row < len(rows) and lead < width:
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


def _in_rowspace(bits: int, basis: tuple[int, ...]) -> bool:
    leftover = bits
    for row in basis:
        pivot = row & -row
        if leftover & pivot:
            leftover ^= row
    return leftover == 0


@dataclass(frozen=True)
class StabilizerCode:
    """A phase-free stabilizer code over an explicit set of data qubits."""

    data_qubits: tuple[int, ...]
    stabilizers: tuple[Pauli, ...]
    logical_x: Pauli
    logical_z: Pauli

    def __post_init__(self) -> None:
        if not self.data_qubits:
            raise ValueError("a stabilizer code needs at least one data qubit")
        if len(set(self.data_qubits)) != len(self.data_qubits):
            raise ValueError("data qubits must be unique")

        for name, pauli in (
            *((f"stabilizer {i}", stabilizer) for i, stabilizer in enumerate(self.stabilizers)),
            ("logical X", self.logical_x),
            ("logical Z", self.logical_z),
        ):
            self.validate_data_pauli(pauli, name=name)

        for i, left in enumerate(self.stabilizers):
            for right in self.stabilizers[i + 1 :]:
                if not left.commutes(right):
                    raise ValueError("stabilizer generators must commute")
        if not all(self.logical_x.commutes(stabilizer) for stabilizer in self.stabilizers):
            raise ValueError("logical X must commute with every stabilizer")
        if not all(self.logical_z.commutes(stabilizer) for stabilizer in self.stabilizers):
            raise ValueError("logical Z must commute with every stabilizer")
        if self.logical_x.commutes(self.logical_z):
            raise ValueError("logical X and logical Z must anticommute")

    @property
    def n(self) -> int:
        return len(self.data_qubits)

    @property
    def syndrome_size(self) -> int:
        return len(self.stabilizers)

    @cached_property
    def _positions(self) -> dict[int, int]:
        return {qubit: position for position, qubit in enumerate(self.data_qubits)}

    def validate_data_pauli(self, pauli: Pauli, *, name: str = "pauli") -> None:
        outside = (pauli.x | pauli.z) - frozenset(self.data_qubits)
        if outside:
            raise ValueError(f"{name} acts outside the data qubits: {sorted(outside)!r}")

    def to_symplectic(self, pauli: Pauli) -> int:
        self.validate_data_pauli(pauli)
        bits = 0
        for qubit in pauli.x:
            bits |= 1 << self._positions[qubit]
        for qubit in pauli.z:
            bits |= 1 << (self.n + self._positions[qubit])
        return bits

    def from_symplectic(self, bits: int) -> Pauli:
        if bits < 0 or bits >= 1 << (2 * self.n):
            raise ValueError(f"symplectic value does not fit {2 * self.n} bits: {bits!r}")
        return Pauli(
            x=frozenset(
                qubit for position, qubit in enumerate(self.data_qubits) if bits & (1 << position)
            ),
            z=frozenset(
                qubit
                for position, qubit in enumerate(self.data_qubits)
                if bits & (1 << (self.n + position))
            ),
        )

    @cached_property
    def stabilizer_basis(self) -> tuple[int, ...]:
        rows = [self.to_symplectic(stabilizer) for stabilizer in self.stabilizers]
        return tuple(_independent_rows(rows, 2 * self.n))

    def stabilizer_rank(self) -> int:
        return len(self.stabilizer_basis)

    def in_stabilizer_group(self, pauli: Pauli) -> bool:
        return _in_rowspace(self.to_symplectic(pauli), self.stabilizer_basis)

    def syndrome(self, error: Pauli) -> tuple[int, ...]:
        self.validate_data_pauli(error, name="error")
        return tuple(0 if error.commutes(stabilizer) else 1 for stabilizer in self.stabilizers)

    def is_logical(self, pauli: Pauli) -> bool:
        self.validate_data_pauli(pauli)
        return all(pauli.commutes(stabilizer) for stabilizer in self.stabilizers) and not (
            self.in_stabilizer_group(pauli)
        )

    def distance(self) -> int:
        best = self.n
        for bits in range(1, 1 << (2 * self.n)):
            pauli = self.from_symplectic(bits)
            weight = pauli.weight()
            if weight >= best:
                continue
            if self.is_logical(pauli):
                best = weight
        return best

    def parameters(self) -> tuple[int, int, int]:
        k = self.n - self.stabilizer_rank()
        return (self.n, k, self.distance())
