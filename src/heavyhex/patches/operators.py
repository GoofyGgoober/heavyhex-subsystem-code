"""Gauges, stabilizers and logicals of the heavy-hex code at odd distance.

Follows Sundaresan et al. 2023, eqs. 1-4, extended to any d as in Chamberland
et al. 2020. Qubit labels are 1-based and run down the columns; Pauli ids are label - 1.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from ..core import Pauli
from ..core.subsystem import SubsystemCode


def label(row: int, column: int, distance: int) -> int:
    """1-based code-qubit label at (row, column), both 1-based."""
    return (column - 1) * distance + row


@dataclass(frozen=True)
class HeavyHexOperators:
    """Supports as 1-based labels. .code holds the same operators as 0-based Paulis."""

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
        """X stabilizers then Z, the same order as code.stabilizers."""
        return tuple(self.x_stabilizers) + tuple(self.z_stabilizers)

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

    @cached_property
    def stabilizer_gauge_factors(self) -> dict[str, tuple[str, ...]]:
        """The gauges whose product is each stabilizer: the same-basis gauges inside it."""
        factors: dict[str, tuple[str, ...]] = {}
        for gauges, stabilizers in (
            (self.x_gauges, self.x_stabilizers),
            (self.z_gauges, self.z_stabilizers),
        ):
            for name, support in stabilizers.items():
                factors[name] = tuple(g for g, s in gauges.items() if set(s) <= set(support))
        return factors

    def _pauli(self, basis: str, support: tuple[int, ...]) -> Pauli:
        ids = frozenset(q - 1 for q in support)
        return Pauli.x_on(ids) if basis == "X" else Pauli.z_on(ids)


def build_operators(distance: int) -> HeavyHexOperators:
    """All operators for an odd distance of at least 3."""
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


def _validate_distance(distance: int) -> int:
    odd_int = isinstance(distance, int) and not isinstance(distance, bool) and distance % 2 == 1
    if not odd_int or distance < 3:
        raise ValueError("distance must be an odd integer of at least 3")
    return distance


def _block(row: int, column: int, distance: int) -> tuple[int, int, int, int]:
    return (
        label(row, column, distance),
        label(row + 1, column, distance),
        label(row, column + 1, distance),
        label(row + 1, column + 1, distance),
    )


def _name(basis: str, support: tuple[int, ...]) -> str:
    return "".join(f"{basis}{q}" for q in support)


D3 = build_operators(3)
D5 = build_operators(5)
