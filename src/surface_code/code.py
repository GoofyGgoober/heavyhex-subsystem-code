from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Iterable, Literal

Pauli = Literal["X", "Y", "Z"]


@dataclass(frozen=True)
class Stabilizer:
    kind: Literal["X", "Z"]
    qubits: frozenset[int]


@dataclass(frozen=True)
class PauliError:
    """Pauli error represented by its binary symplectic X and Z supports."""

    x: frozenset[int] = frozenset()
    z: frozenset[int] = frozenset()

    @classmethod
    def from_ops(cls, operations: Iterable[tuple[int, Pauli]]) -> "PauliError":
        x: set[int] = set()
        z: set[int] = set()
        for qubit, op in operations:
            if op in ("X", "Y"):
                x.symmetric_difference_update((qubit,))
            if op in ("Z", "Y"):
                z.symmetric_difference_update((qubit,))
        return cls(frozenset(x), frozenset(z))


class RotatedSurfaceCode:
    """Distance-d rotated planar CSS code encoding one logical qubit."""

    def __init__(self, distance: int):
        if distance < 3 or distance % 2 == 0:
            raise ValueError("distance must be an odd integer >= 3")
        self.distance = distance
        self.n_data_qubits = distance * distance
        self.x_stabilizers, self.z_stabilizers = self._build_stabilizers()

    def qubit(self, row: int, column: int) -> int:
        return row * self.distance + column

    def _build_stabilizers(self) -> tuple[tuple[Stabilizer, ...], tuple[Stabilizer, ...]]:
        d = self.distance
        xs: list[Stabilizer] = []
        zs: list[Stabilizer] = []
        for row in range(d - 1):
            for col in range(d - 1):
                support = frozenset(
                    self.qubit(r, c)
                    for r, c in ((row, col), (row + 1, col), (row, col + 1), (row + 1, col + 1))
                )
                target = xs if (row + col) % 2 else zs
                target.append(Stabilizer("X" if target is xs else "Z", support))

        # Alternating weight-two checks form the four rough/smooth boundaries.
        for col in range(0, d - 1, 2):
            xs.append(Stabilizer("X", frozenset((self.qubit(0, col), self.qubit(0, col + 1)))))
        for col in range(1, d - 1, 2):
            xs.append(Stabilizer("X", frozenset((self.qubit(d - 1, col), self.qubit(d - 1, col + 1)))))
        for row in range(1, d - 1, 2):
            zs.append(Stabilizer("Z", frozenset((self.qubit(row, 0), self.qubit(row + 1, 0)))))
        for row in range(0, d - 1, 2):
            zs.append(Stabilizer("Z", frozenset((self.qubit(row, d - 1), self.qubit(row + 1, d - 1)))))
        return tuple(xs), tuple(zs)

    @property
    def stabilizers(self) -> tuple[Stabilizer, ...]:
        return self.x_stabilizers + self.z_stabilizers

    def syndrome(self, error: PauliError) -> tuple[int, ...]:
        return tuple(
            len(stabilizer.qubits & (error.z if stabilizer.kind == "X" else error.x)) % 2
            for stabilizer in self.stabilizers
        )

    def sample_depolarizing_error(self, probability: float, rng: random.Random | None = None) -> PauliError:
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        rng = rng or random.Random()
        operations = []
        for qubit in range(self.n_data_qubits):
            if rng.random() < probability:
                operations.append((qubit, rng.choice(("X", "Y", "Z"))))
        return PauliError.from_ops(operations)

    def logical_x_parity(self, error: PauliError) -> int:
        return len(error.z & frozenset(self.qubit(row, 0) for row in range(self.distance))) % 2

    def logical_z_parity(self, error: PauliError) -> int:
        return len(error.x & frozenset(self.qubit(0, col) for col in range(self.distance))) % 2

