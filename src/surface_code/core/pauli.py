"""n-qubit Paulis, up to phase.

A Pauli is a pair of bit-sets: X support and Z support. A qubit in both
is Y. Phase is dropped — we only need commutation and the stabilizer
group as a vector space over GF(2).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


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
