"""Geometry models shared by concrete surface-code patches."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Literal

from ..core import Pauli, StabilizerCode

CheckBasis = Literal["X", "Z"]


@dataclass(frozen=True)
class Check:
    basis: CheckBasis
    support: frozenset[int]
    ancilla: int

    def __post_init__(self) -> None:
        if self.basis not in ("X", "Z"):
            raise ValueError(f"check basis must be X or Z, got {self.basis!r}")

    @property
    def stabilizer(self) -> Pauli:
        if self.basis == "X":
            return Pauli.x_on(self.support)
        return Pauli.z_on(self.support)


@dataclass(frozen=True)
class SurfaceCodePatch:
    distance: int
    data_qubits: tuple[int, ...]
    checks: tuple[Check, ...]
    logical_x: Pauli
    logical_z: Pauli

    def __post_init__(self) -> None:
        if self.distance <= 0:
            raise ValueError("distance must be positive")
        if len(self.data_qubits) != self.distance**2:
            raise ValueError("a rotated patch must have distance squared data qubits")
        if len(set(self.data_qubits)) != len(self.data_qubits):
            raise ValueError("data qubits must be unique")

        ancillas = self.ancillas
        if len(set(ancillas)) != len(ancillas):
            raise ValueError("each check must have a unique ancilla")
        if set(ancillas) & set(self.data_qubits):
            raise ValueError("data qubits and ancillas must be disjoint")

        data = set(self.data_qubits)
        for check in self.checks:
            if not check.support:
                raise ValueError("a check cannot have empty support")
            if not check.support <= data:
                raise ValueError("check support must contain only data qubits")

        # Constructing the algebraic model validates stabilizers and logicals.
        self.code

    @cached_property
    def code(self) -> StabilizerCode:
        return StabilizerCode(
            data_qubits=self.data_qubits,
            stabilizers=self.stabilizers,
            logical_x=self.logical_x,
            logical_z=self.logical_z,
        )

    @property
    def x_checks(self) -> tuple[frozenset[int], ...]:
        return tuple(check.support for check in self.checks if check.basis == "X")

    @property
    def z_checks(self) -> tuple[frozenset[int], ...]:
        return tuple(check.support for check in self.checks if check.basis == "Z")

    @property
    def x_ancillas(self) -> tuple[int, ...]:
        return tuple(check.ancilla for check in self.checks if check.basis == "X")

    @property
    def z_ancillas(self) -> tuple[int, ...]:
        return tuple(check.ancilla for check in self.checks if check.basis == "Z")

    @property
    def ancillas(self) -> tuple[int, ...]:
        return tuple(check.ancilla for check in self.checks)

    @property
    def x_stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks if check.basis == "X")

    @property
    def z_stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks if check.basis == "Z")

    @property
    def stabilizers(self) -> tuple[Pauli, ...]:
        return tuple(check.stabilizer for check in self.checks)

    def data_qubit(self, row: int, col: int) -> int:
        if not (0 <= row < self.distance and 0 <= col < self.distance):
            raise ValueError(f"no data qubit at ({row}, {col})")
        position = row * self.distance + col
        try:
            return self.data_qubits[position]
        except IndexError as error:
            raise ValueError(f"no data qubit at ({row}, {col})") from error
