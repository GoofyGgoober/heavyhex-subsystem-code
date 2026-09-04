"""Repeated syndrome rounds for logical-memory experiments.

The experiment is not tied to a hardware vendor.  Its current runner uses Aer,
while the circuit structure already exposes the operations a hardware adapter
will need: mid-circuit ancilla measurement, reset before reuse, and a separate
classical result for every round.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..core import Pauli
from ..patches import PATCH
from .aer import (
    NUM_QUBITS,
    _append_syndrome_round,
    _apply_error,
    _bits_le,
    _logical_zero_circuit,
    _validate_aer_options,
    shot_z_success,
)

Syndrome = tuple[int, ...]
DataBits = tuple[int, ...]


def _probability(name: str, value: float) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be a probability in [0, 1], got {value!r}")


@dataclass(frozen=True)
class CircuitNoise:
    """Simple circuit-level Pauli and readout noise parameters.

    These are experiment inputs rather than backend calibration data.  The
    model omits topology, idle relaxation, leakage, crosstalk, and correlated
    errors.
    """

    single_qubit: float = 0.0
    two_qubit: float = 0.0
    readout: float = 0.0
    reset: float = 0.0

    def __post_init__(self) -> None:
        _probability("single_qubit", self.single_qubit)
        _probability("two_qubit", self.two_qubit)
        _probability("readout", self.readout)
        _probability("reset", self.reset)

    @classmethod
    def ideal(cls) -> CircuitNoise:
        return cls()

    @property
    def is_ideal(self) -> bool:
        return not any((self.single_qubit, self.two_qubit, self.readout, self.reset))


# A reproducible project baseline, not a model of any particular device.
BASELINE_NOISE = CircuitNoise(
    single_qubit=0.001,
    two_qubit=0.01,
    readout=0.02,
    reset=0.01,
)


@dataclass(frozen=True)
class MemoryShot:
    """One shot: every measured syndrome followed by the final data readout."""

    syndromes: tuple[Syndrome, ...]
    data_bits: DataBits

    def __post_init__(self) -> None:
        width = len(PATCH.ancillas)
        for syndrome in self.syndromes:
            if len(syndrome) != width or any(bit not in (0, 1) for bit in syndrome):
                raise ValueError(f"each syndrome must be {width} binary bits, got {syndrome!r}")
        data_width = len(PATCH.data_qubits)
        if len(self.data_bits) != data_width or any(bit not in (0, 1) for bit in self.data_bits):
            raise ValueError(
                f"data readout must be {data_width} binary bits, got {self.data_bits!r}"
            )

    def detection_events(self) -> tuple[Syndrome, ...]:
        """Changes between adjacent syndromes, using all-zero as the initial boundary."""
        previous = (0,) * len(PATCH.ancillas)
        events: list[Syndrome] = []
        for syndrome in self.syndromes:
            events.append(tuple(before ^ after for before, after in zip(previous, syndrome)))
            previous = syndrome
        return tuple(events)


@dataclass(frozen=True)
class MemorySummary:
    rounds: int
    shots: int
    syndrome_trigger_rate: tuple[float, ...]
    detection_event_rate: tuple[float, ...]
    raw_z_success_rate: float
    last_round_z_success_rate: float | None


def _validate_rounds(rounds: int) -> None:
    if not isinstance(rounds, int) or isinstance(rounds, bool) or rounds < 0:
        raise ValueError(f"rounds must be a non-negative integer, got {rounds!r}")


def to_memory_circuit(rounds: int, error: Pauli | None = None):
    """Prepare logical zero, extract ``rounds`` syndromes, then measure data in Z."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    _validate_rounds(rounds)
    qubits = QuantumRegister(NUM_QUBITS, "q")
    syndrome_registers = [
        ClassicalRegister(len(PATCH.ancillas), f"syn_{round_index}")
        for round_index in range(rounds)
    ]
    data = ClassicalRegister(len(PATCH.data_qubits), "data")
    circuit = QuantumCircuit(qubits, *syndrome_registers, data)
    circuit.compose(_logical_zero_circuit(), PATCH.data_qubits, inplace=True)
    if error is not None:
        _apply_error(circuit, error)

    for syndrome_register in syndrome_registers:
        circuit.reset(PATCH.ancillas)
        _append_syndrome_round(circuit, syndrome_register)
        circuit.barrier()

    for index, qubit in enumerate(PATCH.data_qubits):
        circuit.measure(qubit, data[index])
    return circuit


def _noise_model(noise: CircuitNoise):
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error

    model = NoiseModel()
    if noise.single_qubit:
        one_qubit_error = depolarizing_error(noise.single_qubit, 1)
        model.add_all_qubit_quantum_error(
            one_qubit_error,
            ("h", "s", "sdg", "x", "y", "z"),
        )
    if noise.two_qubit:
        model.add_all_qubit_quantum_error(depolarizing_error(noise.two_qubit, 2), ("cx",))
    if noise.readout:
        readout_error = ReadoutError(
            [[1 - noise.readout, noise.readout], [noise.readout, 1 - noise.readout]]
        )
        model.add_all_qubit_readout_error(readout_error)
    if noise.reset:
        reset_error = pauli_error((("X", noise.reset), ("I", 1 - noise.reset)))
        model.add_all_qubit_quantum_error(reset_error, ("reset",))
    return model


def _parse_memory_shot(key: str, rounds: int) -> MemoryShot:
    parts = key.split()
    if len(parts) != rounds + 1:
        raise ValueError(f"expected {rounds + 1} classical registers, got {key!r}")
    data_bits = _bits_le(parts[0])
    syndromes = tuple(_bits_le(bits) for bits in reversed(parts[1:]))
    return MemoryShot(syndromes=syndromes, data_bits=data_bits)


def run_memory(
    rounds: int,
    *,
    shots: int = 1024,
    seed: int | None = None,
    error: Pauli | None = None,
    noise: CircuitNoise | None = None,
) -> dict[MemoryShot, int]:
    """Run a repeated-round memory circuit and tally complete shot histories."""
    _validate_rounds(rounds)
    _validate_aer_options(shots, seed)
    noise = BASELINE_NOISE if noise is None else noise
    if not isinstance(noise, CircuitNoise):
        raise TypeError(f"noise must be CircuitNoise, got {type(noise).__name__}")

    from qiskit_aer import AerSimulator

    circuit = to_memory_circuit(rounds, error)
    backend_options: dict[str, object] = {"method": "stabilizer"}
    if not noise.is_ideal:
        backend_options["noise_model"] = _noise_model(noise)
    run_options: dict[str, int] = {"shots": shots}
    if seed is not None:
        run_options["seed_simulator"] = seed
    result = AerSimulator(**backend_options).run(circuit, **run_options).result()

    tallies: dict[MemoryShot, int] = {}
    for key, count in result.get_counts().items():
        parsed = _parse_memory_shot(key, rounds)
        tallies[parsed] = tallies.get(parsed, 0) + count
    return tallies


def summarize_memory(tallies: dict[MemoryShot, int]) -> MemorySummary:
    """Summarize raw histories without claiming space-time decoding."""
    if not tallies:
        raise ValueError("memory experiment returned no outcomes")
    first = next(iter(tallies))
    rounds = len(first.syndromes)
    shots = 0
    syndrome_triggers = [0] * rounds
    detection_events = [0] * rounds
    raw_successes = 0
    last_round_successes = 0

    for shot, count in tallies.items():
        if len(shot.syndromes) != rounds:
            raise ValueError("memory outcomes have inconsistent round counts")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            raise ValueError(f"outcome count must be a positive integer, got {count!r}")
        shots += count
        for round_index, syndrome in enumerate(shot.syndromes):
            syndrome_triggers[round_index] += count * int(any(syndrome))
        for round_index, events in enumerate(shot.detection_events()):
            detection_events[round_index] += count * sum(events)
        raw_successes += count * int(
            sum(shot.data_bits[qubit] for qubit in PATCH.logical_z.z) % 2 == 0
        )
        if rounds:
            last_round_successes += count * shot_z_success(
                shot.syndromes[-1], shot.data_bits
            )

    return MemorySummary(
        rounds=rounds,
        shots=shots,
        syndrome_trigger_rate=tuple(value / shots for value in syndrome_triggers),
        detection_event_rate=tuple(
            value / (shots * len(PATCH.ancillas)) for value in detection_events
        ),
        raw_z_success_rate=raw_successes / shots,
        last_round_z_success_rate=last_round_successes / shots if rounds else None,
    )
