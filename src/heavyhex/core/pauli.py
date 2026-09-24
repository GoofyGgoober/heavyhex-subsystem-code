"""Paulis without phases, stored as their X and Z supports. A qubit in both has a Y."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class Pauli:
    x: frozenset[int] = frozenset()
    z: frozenset[int] = frozenset()

    @classmethod
    def x_on(cls, qubits: Iterable[int]) -> Pauli:
        return cls(x=frozenset(qubits))

    @classmethod
    def z_on(cls, qubits: Iterable[int]) -> Pauli:
        return cls(z=frozenset(qubits))

    def __mul__(self, other: Pauli) -> Pauli:
        return Pauli(self.x ^ other.x, self.z ^ other.z)

    def commutes(self, other: Pauli) -> bool:
        return (len(self.x & other.z) + len(self.z & other.x)) % 2 == 0

    def weight(self) -> int:
        return len(self.x | self.z)
