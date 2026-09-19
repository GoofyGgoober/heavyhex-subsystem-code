"""Abstract heavy-hex gauge circuits: one fresh ancilla per gauge measurement.

No flags or relays here; each gauge is measured directly (H, CX/CZ chain, H,
measure). This validates code semantics in Aer: checks frozen, gauges random,
logical preserved, faults correctable. Device-faithful flagged circuits with
explicit CX order live in heavyhex_flagged.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core import Pauli
from ..patches.heavyhex import D3, HeavyHexOperators

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


@dataclass(frozen=True)
class GaugeSlot:
    round: int  # -1 for the prep half, 0..rounds-1 otherwise
    half: str  # "X" or "Z"
    name: str
    bit: int  # index into the gauge classical register


@dataclass(frozen=True)
class MemorySchedule:
    patch: HeavyHexOperators
    basis: str
    rounds: int
    gauges: tuple[GaugeSlot, ...]

    def outcomes(self, gauge_bits: tuple[int, ...]) -> dict[tuple[int, str, str], int]:
        return {(slot.round, slot.half, slot.name): gauge_bits[slot.bit] for slot in self.gauges}

    def checks(self, gauge_bits: tuple[int, ...]) -> dict[tuple[int, str], int]:
        """Derived stabilizer values per round from gauge products."""
        outcomes = self.outcomes(gauge_bits)
        factors = self.patch.stabilizer_gauge_factors()
        checks: dict[tuple[int, str], int] = {}
        rounds = sorted({slot.round for slot in self.gauges})
        for stab_name, gauge_names in factors.items():
            half = "X" if stab_name.startswith("X") else "Z"
            for round in rounds:
                try:
                    bits = [outcomes[round, half, name] for name in gauge_names]
                except KeyError:
                    continue  # prep half only measures one basis
                checks[round, stab_name] = sum(bits) % 2
        return checks


def _supports(patch: HeavyHexOperators, half: str) -> list[tuple[str, tuple[int, ...]]]:
    gauges = patch.x_gauges if half == "X" else patch.z_gauges
    return [(name, tuple(q - 1 for q in support)) for name, support in gauges.items()]


def _measure_gauge(circuit: QuantumCircuit, half: str, data: tuple[int, ...], ancilla: int) -> None:
    circuit.h(ancilla)
    for qubit in data:
        if half == "X":
            circuit.cx(ancilla, qubit)
        else:
            circuit.cz(ancilla, qubit)
    circuit.h(ancilla)


def _apply_pauli(circuit: QuantumCircuit, error: Pauli) -> None:
    for qubit in error.x - error.z:
        circuit.x(qubit)
    for qubit in error.z - error.x:
        circuit.z(qubit)
    for qubit in error.x & error.z:
        circuit.y(qubit)


def memory_circuit(
    patch: HeavyHexOperators = D3,
    *,
    rounds: int = 1,
    basis: str = "Z",
    error: Pauli | None = None,
    inject_at: str = "after_prep",
) -> tuple[QuantumCircuit, MemorySchedule]:
    """Prep |basis>_L, run rounds of gauge halves, read out the data.

    Z basis: prep |0>^n + X half, then rounds of (Z half, X half), Z readout.
    X basis: prep |+>^n + Z half, then rounds of (X half, Z half), X readout.
    inject_at selects the stage boundary for the optional Pauli error.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    if basis not in ("X", "Z"):
        raise ValueError(f"basis must be X or Z, got {basis!r}")
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 1:
        raise ValueError(f"rounds must be a positive integer, got {rounds!r}")

    prep_half = "X" if basis == "Z" else "Z"
    halves = ("Z", "X") if basis == "Z" else ("X", "Z")
    stages = ["after_prep"] + [f"after_{half.lower()}{r}" for r in range(rounds) for half in halves]
    if inject_at not in stages:
        raise ValueError(f"inject_at must be one of {stages}, got {inject_at!r}")
    if error is not None:
        patch.code.validate_data_pauli(error, name="error")

    n = patch.distance * patch.distance
    gauge_names = [("X", name) for name in patch.x_gauges] + [
        ("Z", name) for name in patch.z_gauges
    ]
    total = len(_supports(patch, prep_half)) + rounds * len(gauge_names)
    qubits = QuantumRegister(n + total, "q")
    gauge = ClassicalRegister(total, "g")
    data = ClassicalRegister(n, "d")
    circuit = QuantumCircuit(qubits, gauge, data)

    if basis == "X":
        circuit.h(list(range(n)))

    slots: list[GaugeSlot] = []
    ancilla = n
    bit = 0

    def half_round(round: int, half: str) -> None:
        nonlocal ancilla, bit
        for name, support in _supports(patch, half):
            _measure_gauge(circuit, half, support, ancilla)
            circuit.measure(ancilla, gauge[bit])
            slots.append(GaugeSlot(round, half, name, bit))
            ancilla += 1
            bit += 1

    def maybe_inject(stage: str) -> None:
        if error is not None and inject_at == stage:
            _apply_pauli(circuit, error)

    half_round(-1, prep_half)
    circuit.barrier()
    maybe_inject("after_prep")
    for r in range(rounds):
        for half in halves:
            half_round(r, half)
            circuit.barrier()
            maybe_inject(f"after_{half.lower()}{r}")

    if basis == "X":
        circuit.h(list(range(n)))
    for i in range(n):
        circuit.measure(i, data[i])

    return circuit, MemorySchedule(patch, basis, rounds, tuple(slots))
