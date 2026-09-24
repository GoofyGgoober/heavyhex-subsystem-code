"""Textbook gauge circuits: a fresh ancilla for every gauge measurement, no flags."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core import Pauli
from ..patches.operators import D3, HeavyHexOperators

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


@dataclass(frozen=True)
class GaugeSlot:
    round: int  # -1 for the prep half, 0..rounds-1 otherwise
    half: str  # "X" or "Z"
    gauge: str
    bit: int  # index into the gauge classical register


@dataclass(frozen=True)
class MemorySchedule:
    patch: HeavyHexOperators
    basis: str
    rounds: int
    measurements: tuple[GaugeSlot, ...]

    def checks(self, gauge_bits: tuple[int, ...]) -> dict[tuple[int, str], int]:
        """Each stabilizer's value per round, the XOR of its gauge outcomes."""
        outcomes = {(m.round, m.half, m.gauge): gauge_bits[m.bit] for m in self.measurements}
        return _checks(self.patch, outcomes)


def memory_circuit(
    patch: HeavyHexOperators = D3,
    *,
    rounds: int = 1,
    basis: str = "Z",
    error: Pauli | None = None,
    inject_at: str = "after_prep",
) -> tuple[QuantumCircuit, MemorySchedule]:
    """Prepare logical |0> (or |+>), measure the gauges for some rounds, read out the data.

    In the Z basis: |0> on every data qubit, an X half, then each round a Z half
    and an X half, then Z readout. The X basis swaps X and Z.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    prep_half, halves = _halves(patch, rounds, basis, error, inject_at)

    n = patch.distance * patch.distance
    total = len(_supports(patch, prep_half)) + rounds * (len(patch.x_gauges) + len(patch.z_gauges))
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

    half_round(-1, prep_half)
    circuit.barrier()
    if inject_at == "after_prep":
        _apply_pauli(circuit, error)
    for r in range(rounds):
        for half in halves:
            half_round(r, half)
            circuit.barrier()
            if inject_at == f"after_{half.lower()}{r}":
                _apply_pauli(circuit, error)

    if basis == "X":
        circuit.h(list(range(n)))
    for i in range(n):
        circuit.measure(i, data[i])

    return circuit, MemorySchedule(patch, basis, rounds, tuple(slots))


def _halves(
    patch: HeavyHexOperators,
    rounds: int,
    basis: str,
    error: Pauli | None,
    inject_at: str,
) -> tuple[str, tuple[str, str]]:
    """Check the options; return the prep half and the order of halves in a round."""
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
    return prep_half, halves


def _checks(
    patch: HeavyHexOperators, outcomes: dict[tuple[int, str, str], int]
) -> dict[tuple[int, str], int]:
    """XOR each stabilizer's gauge outcomes, round by round.

    outcomes is keyed by (round, half, gauge). The prep half measures only one
    basis, so round -1 has only those stabilizers.
    """
    checks: dict[tuple[int, str], int] = {}
    rounds = sorted({round for round, _, _ in outcomes})
    for stab_name, gauge_names in patch.stabilizer_gauge_factors.items():
        half = "X" if stab_name.startswith("X") else "Z"
        for round in rounds:
            try:
                bits = [outcomes[round, half, name] for name in gauge_names]
            except KeyError:
                continue
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


def _apply_pauli(circuit: QuantumCircuit, error: Pauli | None) -> None:
    if error is None:
        return
    for qubit in error.x - error.z:
        circuit.x(qubit)
    for qubit in error.z - error.x:
        circuit.z(qubit)
    for qubit in error.x & error.z:
        circuit.y(qubit)
