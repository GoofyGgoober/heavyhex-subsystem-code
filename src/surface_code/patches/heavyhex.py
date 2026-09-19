"""Heavy-hex subsystem code: gauge/stabilizer/logical operators for odd distance.

Column-major code-qubit labels 1..d*d (Sundaresan et al. 2023, eqs. 1-4,
generalized via Chamberland et al. 2020). Data-qubit ids are 0-based
(label - 1), matching the rest of this package. Device embedding lives in
heavyhex_embed.py; this module is pure code mathematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from ..core import Pauli
from ..core.subsystem import SubsystemCode


def _validate_distance(distance: int) -> int:
    if isinstance(distance, bool) or not isinstance(distance, int):
        raise ValueError("distance must be an odd integer of at least 3")
    if distance < 3 or distance % 2 == 0:
        raise ValueError("distance must be an odd integer of at least 3")
    return distance


def label(row: int, column: int, distance: int) -> int:
    """1-based code-qubit label at (row, column), both 1-based."""
    return (column - 1) * distance + row


def _block(row: int, column: int, distance: int) -> tuple[int, int, int, int]:
    return (
        label(row, column, distance),
        label(row + 1, column, distance),
        label(row, column + 1, distance),
        label(row + 1, column + 1, distance),
    )


def _name(basis: str, support: tuple[int, ...]) -> str:
    return "".join(f"{basis}{q}" for q in sorted(support))


def _solve_product(supports: list[set[int]], target: set[int]) -> list[int] | None:
    """Subset of supports whose symmetric difference is target, or None."""
    universe = sorted(set().union(*supports, target))
    matrix = [[1 if q in support else 0 for support in supports] for q in universe]
    rhs = [1 if q in target else 0 for q in universe]
    pivot_row: dict[int, int] = {}
    row = 0
    for column in range(len(supports)):
        pivot = next((i for i in range(row, len(matrix)) if matrix[i][column]), None)
        if pivot is None:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        rhs[row], rhs[pivot] = rhs[pivot], rhs[row]
        pivot_row[column] = row
        for i in range(len(matrix)):
            if i != row and matrix[i][column]:
                matrix[i] = [a ^ b for a, b in zip(matrix[i], matrix[row])]
                rhs[i] ^= rhs[row]
        row += 1
    for i in range(len(matrix)):
        if all(bit == 0 for bit in matrix[i]) and rhs[i]:
            return None
    return [rhs[pivot_row[column]] if column in pivot_row else 0 for column in range(len(supports))]


@dataclass(frozen=True)
class HeavyHexOperators:
    """Labeled operator supports (1-based labels) plus Pauli forms (0-based)."""

    distance: int
    x_gauges: dict[str, tuple[int, ...]]
    z_gauges: dict[str, tuple[int, ...]]
    x_stabilizers: dict[str, tuple[int, ...]]
    z_stabilizers: dict[str, tuple[int, ...]]
    logical_x: tuple[int, ...]
    logical_z: tuple[int, ...]

    @property
    def data_qubits(self) -> tuple[int, ...]:
        return tuple(range(self.distance * self.distance))

    @property
    def stabilizer_names(self) -> tuple[str, ...]:
        """Names in SubsystemCode.stabilizers order (X stabilizers, then Z)."""
        return tuple(self.x_stabilizers) + tuple(self.z_stabilizers)

    def _pauli(self, basis: str, support: tuple[int, ...]) -> Pauli:
        ids = frozenset(q - 1 for q in support)
        return Pauli.x_on(ids) if basis == "X" else Pauli.z_on(ids)

    @cached_property
    def code(self) -> SubsystemCode:
        return SubsystemCode(
            data_qubits=self.data_qubits,
            gauge_x=tuple(self._pauli("X", s) for s in self.x_gauges.values()),
            gauge_z=tuple(self._pauli("Z", s) for s in self.z_gauges.values()),
            stabilizers=tuple(
                [self._pauli("X", s) for s in self.x_stabilizers.values()]
                + [self._pauli("Z", s) for s in self.z_stabilizers.values()]
            ),
            logical_x=self._pauli("X", self.logical_x),
            logical_z=self._pauli("Z", self.logical_z),
        )

    def stabilizer_gauge_factors(self) -> dict[str, tuple[str, ...]]:
        """Each stabilizer as a product of same-basis gauges (GF(2) solve)."""
        factors: dict[str, tuple[str, ...]] = {}
        for gauges, stabilizers in (
            (self.x_gauges, self.x_stabilizers),
            (self.z_gauges, self.z_stabilizers),
        ):
            names = list(gauges)
            supports = [set(gauges[name]) for name in names]
            for stab_name, support in stabilizers.items():
                solution = _solve_product(supports, set(support))
                if solution is None:
                    raise RuntimeError(f"{stab_name} is not a product of same-basis gauges")
                factors[stab_name] = tuple(name for name, bit in zip(names, solution) if bit)
        return factors


def build_operators(distance: int) -> HeavyHexOperators:
    """Gauge set, stabilizers and logicals for odd distance >= 3."""
    d = _validate_distance(distance)
    x_gauges: dict[str, tuple[int, ...]] = {}
    z_gauges: dict[str, tuple[int, ...]] = {}
    x_stabilizers: dict[str, tuple[int, ...]] = {}
    z_stabilizers: dict[str, tuple[int, ...]] = {}

    for column in range(1, d):
        for row in range(1, d + 1):
            support = (label(row, column, d), label(row, column + 1, d))
            x_gauges[_name("X", support)] = support

    for row in range(1, d):
        for column in range(1, d):
            support = _block(row, column, d)
            if (row + column) % 2 == 0:
                x_stabilizers[_name("X", support)] = support
            else:
                z_gauges[_name("Z", support)] = support
        # Alternate left and right boundary gauges, as in the d=3 figure.
        column = 1 if row % 2 else d
        support = (label(row, column, d), label(row + 1, column, d))
        z_gauges[_name("Z", support)] = support
        support = tuple(label(r, c, d) for c in range(1, d + 1) for r in (row, row + 1))
        z_stabilizers[_name("Z", support)] = support

    for column in range(1, d):
        row = d if column % 2 else 1
        support = (label(row, column, d), label(row, column + 1, d))
        x_stabilizers[_name("X", support)] = support

    return HeavyHexOperators(
        distance=d,
        x_gauges=x_gauges,
        z_gauges=z_gauges,
        x_stabilizers=x_stabilizers,
        z_stabilizers=z_stabilizers,
        logical_x=tuple(range(1, d + 1)),
        logical_z=tuple(range(1, d * d + 1, d)),
    )


D3 = build_operators(3)
D5 = build_operators(5)

# d=3 gauge-qubit operators: A = (X1X4, Z1Z2), B = (X5X8, Z8Z9), 0-based ids.
GAUGE_QUBITS_D3: tuple[tuple[Pauli, Pauli], ...] = (
    (Pauli.x_on((0, 3)), Pauli.z_on((0, 1))),
    (Pauli.x_on((4, 7)), Pauli.z_on((7, 8))),
)
