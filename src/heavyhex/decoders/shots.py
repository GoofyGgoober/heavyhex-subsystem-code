"""Turn a shot's measurements into detectors for MWPM.

A stabilizer's value in a round is the XOR of its gauge outcomes, and a round's
values are its syndrome. A detector is a stabilizer's value XOR its value the
round before. The last round is compared with the data readout, so it only has
stabilizers of the memory basis.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

from .._validation import validate_binary_bits
from ..circuits.flagged import FlaggedSchedule
from ..circuits.ideal import MemorySchedule
from ..core import Pauli
from ..patches.operators import HeavyHexOperators


@dataclass(frozen=True)
class Memory:
    """A memory run: the patch, the basis it stores, and how many rounds.

    Detectors are listed round by round, in patch.stabilizer_names order.
    """

    patch: HeavyHexOperators
    basis: str
    rounds: int

    def __post_init__(self) -> None:
        if self.basis not in ("X", "Z"):
            raise ValueError("basis must be X or Z")
        if isinstance(self.rounds, bool) or not isinstance(self.rounds, int) or self.rounds < 1:
            raise ValueError("rounds must be a positive integer")

    @cached_property
    def labels(self) -> tuple[tuple[int, str], ...]:
        """(round, stabilizer) of each detector."""
        return tuple(
            (r, name) for r in range(self.rounds) for name in self.patch.stabilizer_names
        ) + tuple(
            (self.rounds, name)
            for name in self.patch.stabilizer_names
            if name.startswith(self.basis)
        )

    @cached_property
    def basis_indices(self) -> tuple[int, ...]:
        """Positions of the detectors on memory-basis stabilizers, the only ones MWPM uses."""
        return tuple(i for i, (_, name) in enumerate(self.labels) if name.startswith(self.basis))

    @cached_property
    def logical_support(self) -> frozenset[int]:
        logical = self.patch.code.logical_x if self.basis == "X" else self.patch.code.logical_z
        return logical.x or logical.z


@dataclass(frozen=True)
class Shot:
    memory: Memory
    detectors: tuple[int, ...]
    data_bits: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_binary_bits("detectors", self.detectors, len(self.memory.labels))
        validate_binary_bits("data readout", self.data_bits, self.memory.patch.code.n)

    @property
    def logical_bit(self) -> int:
        return sum(self.data_bits[q] for q in self.memory.logical_support) % 2

    def logical_error(self, correction: Pauli) -> bool:
        """True if the readout, with `correction` applied, has the wrong logical value."""
        flips = correction.x if self.memory.basis == "Z" else correction.z
        return bool((self.logical_bit + len(flips & self.memory.logical_support)) % 2)


def read_shot(
    schedule: MemorySchedule | FlaggedSchedule,
    gauge_bits: tuple[int, ...],
    data_bits: tuple[int, ...],
) -> Shot:
    """Detectors and data bits for one shot of a memory circuit.

    In the flagged circuit, X readout bits are flipped back where a flag left a Z.
    """
    memory = Memory(schedule.patch, schedule.basis, schedule.rounds)
    validate_binary_bits("gauge outcomes", gauge_bits, len(schedule.measurements))
    validate_binary_bits("data readout", data_bits, schedule.patch.code.n)
    syndromes = schedule.checks(gauge_bits)
    if isinstance(schedule, FlaggedSchedule) and schedule.basis == "X":
        z = Pauli()
        for _, pauli in schedule.z_from_flags(gauge_bits):
            z *= pauli
        data_bits = tuple(bit ^ (q in z.z) for q, bit in enumerate(data_bits))
    return Shot(memory, detectors_from_syndromes(memory, syndromes, data_bits), data_bits)


def detectors_from_syndromes(
    memory: Memory,
    syndromes: dict[tuple[int, str], int],
    data_bits: tuple[int, ...],
) -> tuple[int, ...]:
    """XOR each stabilizer's value with its value the round before.

    Memory-basis stabilizers start at 0. The others start from the prep half
    (round -1), where they come out random. The last round is compared with the
    same stabilizers computed from the data readout.
    """
    validate_binary_bits("data readout", data_bits, memory.patch.code.n)
    prep = "X" if memory.basis == "Z" else "Z"
    required = {(r, n) for r in range(memory.rounds) for n in memory.patch.stabilizer_names}
    required |= {(-1, n) for n in memory.patch.stabilizer_names if n.startswith(prep)}
    if set(syndromes) != required:
        raise ValueError("syndrome history does not match the preparation and round schedule")
    validate_binary_bits(
        "syndrome history", tuple(syndromes[key] for key in sorted(required)), len(required)
    )
    stabs = memory.patch.x_stabilizers if memory.basis == "X" else memory.patch.z_stabilizers
    detectors = []
    for r, name in memory.labels:
        if r == memory.rounds:
            current = sum(data_bits[q - 1] for q in stabs[name]) % 2
        else:
            current = syndromes[r, name]
        previous = syndromes[r - 1, name] if r or name.startswith(prep) else 0
        detectors.append(current ^ previous)
    return tuple(detectors)
