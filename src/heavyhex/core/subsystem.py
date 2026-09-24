"""CSS subsystem codes, with Paulis as GF(2) bitmasks.

G is the gauge group and S, its center, the stabilizers. The distance is the
weight of the lightest Pauli that commutes with S but is not in G.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from functools import cached_property
from itertools import combinations, product

from .pauli import Pauli


@dataclass(frozen=True)
class SubsystemCode:
    """Checks on construction that the stabilizers and logicals fit the gauges."""

    data_qubits: tuple[int, ...]
    gauge_x: tuple[Pauli, ...]
    gauge_z: tuple[Pauli, ...]
    stabilizers: tuple[Pauli, ...]
    logical_x: Pauli
    logical_z: Pauli

    def __post_init__(self) -> None:
        if not self.data_qubits:
            raise ValueError("a subsystem code needs at least one data qubit")
        if len(set(self.data_qubits)) != len(self.data_qubits):
            raise ValueError("data qubits must be unique")

        for name, pauli in (
            *((f"X gauge {i}", g) for i, g in enumerate(self.gauge_x)),
            *((f"Z gauge {i}", g) for i, g in enumerate(self.gauge_z)),
            *((f"stabilizer {i}", s) for i, s in enumerate(self.stabilizers)),
            ("logical X", self.logical_x),
            ("logical Z", self.logical_z),
        ):
            self.validate_data_pauli(pauli, name=name)
        if any(g.z for g in self.gauge_x) or any(g.x for g in self.gauge_z):
            raise ValueError("gauge generators must be pure X or pure Z (CSS)")

        gauges = self.gauge_x + self.gauge_z
        for i, stabilizer in enumerate(self.stabilizers):
            for gauge in gauges:
                if not stabilizer.commutes(gauge):
                    raise ValueError(f"stabilizer {i} is not central in the gauge group")
            if not self.in_gauge_group(stabilizer):
                raise ValueError(f"stabilizer {i} is not a product of gauges")
        for name, logical in (("logical X", self.logical_x), ("logical Z", self.logical_z)):
            for gauge in gauges:
                if not logical.commutes(gauge):
                    raise ValueError(f"{name} must commute with every gauge generator")
            if self.in_gauge_group(logical):
                raise ValueError(f"{name} must not be in the gauge group")
        if self.logical_x.commutes(self.logical_z):
            raise ValueError("logical X and logical Z must anticommute")

    @property
    def n(self) -> int:
        return len(self.data_qubits)

    @property
    def syndrome_size(self) -> int:
        return len(self.stabilizers)

    def validate_data_pauli(self, pauli: Pauli, *, name: str = "pauli") -> None:
        outside = (pauli.x | pauli.z) - frozenset(self.data_qubits)
        if outside:
            raise ValueError(f"{name} acts outside the data qubits: {sorted(outside)!r}")

    def to_symplectic(self, pauli: Pauli) -> int:
        """Bitmask with X support in bits 0..n-1 and Z support in bits n..2n-1."""
        self.validate_data_pauli(pauli)
        bits = 0
        for qubit in pauli.x:
            bits |= 1 << self._positions[qubit]
        for qubit in pauli.z:
            bits |= 1 << (self.n + self._positions[qubit])
        return bits

    @cached_property
    def gauge_basis(self) -> tuple[int, ...]:
        rows = [self.to_symplectic(g) for g in self.gauge_x + self.gauge_z]
        return tuple(_independent_rows(rows, 2 * self.n))

    @cached_property
    def stabilizer_basis(self) -> tuple[int, ...]:
        rows = [self.to_symplectic(s) for s in self.stabilizers]
        return tuple(_independent_rows(rows, 2 * self.n))

    def gauge_rank(self) -> int:
        return len(self.gauge_basis)

    def stabilizer_rank(self) -> int:
        return len(self.stabilizer_basis)

    def gauge_qubit_count(self) -> int:
        return (self.gauge_rank() - self.stabilizer_rank()) // 2

    def logical_count(self) -> int:
        return self.n - self.stabilizer_rank() - self.gauge_qubit_count()

    def in_gauge_group(self, pauli: Pauli) -> bool:
        return _in_rowspace(self.to_symplectic(pauli), self.gauge_basis)

    def syndrome(self, error: Pauli) -> tuple[int, ...]:
        self.validate_data_pauli(error, name="error")
        return tuple(0 if error.commutes(s) else 1 for s in self.stabilizers)

    def is_logical(self, pauli: Pauli) -> bool:
        """Undetectable but not a gauge, so it acts on the logical qubit."""
        return all(pauli.commutes(s) for s in self.stabilizers) and not self.in_gauge_group(pauli)

    def paulis_of_weight(self, weight: int, axes: str = "XYZ") -> Iterator[Pauli]:
        """Every Pauli of this weight using only the given axes.

        The lookup table keeps the first error it sees per syndrome, so this order
        decides its corrections.
        """
        for qubits in combinations(self.data_qubits, weight):
            for letters in product(axes, repeat=weight):
                x = frozenset(q for q, a in zip(qubits, letters) if a in "XY")
                z = frozenset(q for q, a in zip(qubits, letters) if a in "YZ")
                yield Pauli(x, z)

    def distance(self, max_weight: int | None = None) -> int | None:
        """Weight of the lightest logical, or None if there is none up to max_weight."""
        limit = self.n if max_weight is None else max_weight
        for weight in range(1, limit + 1):
            for pauli in self.paulis_of_weight(weight):
                if self.is_logical(pauli):
                    return weight
        return None

    @cached_property
    def _positions(self) -> dict[int, int]:
        return {qubit: position for position, qubit in enumerate(self.data_qubits)}


def _independent_rows(rows: Iterable[int], width: int) -> list[int]:
    """GF(2) row reduction. Each row that comes back has its pivot at its lowest set bit."""
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
    """True when bits is a sum of basis rows. basis must come from _independent_rows."""
    leftover = bits
    for row in basis:
        pivot = row & -row
        if leftover & pivot:
            leftover ^= row
    return leftover == 0
