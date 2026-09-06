"""Repeated syndrome rounds for logical-memory experiments.

The experiment is not tied to a hardware vendor.  Its current runner uses Aer,
while the circuit structure already exposes the operations a hardware adapter
will need: mid-circuit ancilla measurement, reset before reuse, and a separate
classical result for every round.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import expm1
from typing import TYPE_CHECKING

from .._validation import validate_binary_bits, validate_nonnegative_number, validate_probability
from ..core import Pauli
from ..decoders.infer_logical import normalize_measurement_basis
from ..patches import PATCH
from .aer import (
    NUM_QUBITS,
    _append_syndrome_round,
    _apply_error,
    _bits_le,
    _logical_state_circuit,
    _validate_aer_options,
    shot_z_success,
)

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit_aer.noise import NoiseModel

Syndrome = tuple[int, ...]
DataBits = tuple[int, ...]

PREPARATIONS = ("encoder", "product")
"""How the logical state is prepared before the first round.

``encoder``: a synthesized Clifford takes ``|0>^9`` to the exact logical
eigenstate, so every check is deterministic from the first round; its 18
two-qubit gates are noisy and the circuit is not fault-tolerant.
``product``: every data qubit starts in ``|0>`` (``|+>`` for an X-basis test),
as hardware experiments do. The checks of the measured type are deterministic
from the first round; the other type's first outcomes are random and only fix
a reference frame, which the history decoder never reads.
"""


def _validate_preparation(preparation: str) -> None:
    if preparation not in PREPARATIONS:
        choices = ", ".join(PREPARATIONS)
        raise ValueError(f"preparation must be one of {choices}, got {preparation!r}")


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
        validate_probability("single_qubit", self.single_qubit)
        validate_probability("two_qubit", self.two_qubit)
        validate_probability("readout", self.readout)
        validate_probability("reset", self.reset)

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
class IdleNoise:
    """Continuous independent Pauli-event rates per microsecond."""

    x_rate: float = 0.0
    y_rate: float = 0.0
    z_rate: float = 0.0

    def __post_init__(self) -> None:
        validate_nonnegative_number("x_rate", self.x_rate)
        validate_nonnegative_number("y_rate", self.y_rate)
        validate_nonnegative_number("z_rate", self.z_rate)

    @classmethod
    def ideal(cls) -> IdleNoise:
        return cls()

    @property
    def is_ideal(self) -> bool:
        return not any((self.x_rate, self.y_rate, self.z_rate))

    def pauli_probabilities(self, duration_us: float) -> tuple[float, float, float, float]:
        """Return exact (I, X, Y, Z) probabilities after a Poisson interval."""
        validate_nonnegative_number("duration_us", duration_us)
        # An even number of identical Pauli events cancels. expm1 preserves
        # tiny odd-event probabilities without subtracting nearly equal numbers.
        odd_x, odd_y, odd_z = (
            -expm1(-2 * (rate * duration_us)) / 2
            for rate in (self.x_rate, self.y_rate, self.z_rate)
        )
        even_x, even_y, even_z = 1 - odd_x, 1 - odd_y, 1 - odd_z
        # XYZ = I up to phase, so each net Pauli has two parity patterns.
        probability_i = even_x * even_y * even_z + odd_x * odd_y * odd_z
        probability_x = odd_x * even_y * even_z + even_x * odd_y * odd_z
        probability_y = even_x * odd_y * even_z + odd_x * even_y * odd_z
        probability_z = even_x * even_y * odd_z + odd_x * odd_y * even_z
        return probability_i, probability_x, probability_y, probability_z

    def relevant_bit_probability(self, basis: str, duration_us: float) -> float:
        """Probability of the error component visible in an X or Z readout."""
        checked_basis = normalize_measurement_basis(basis)
        _, probability_x, probability_y, probability_z = self.pauli_probabilities(duration_us)
        if checked_basis == "Z":
            return probability_x + probability_y
        return probability_z + probability_y


# Time is expressed in microseconds, but these remain project baselines rather
# than calibration data for any particular device.
BASELINE_IDLE_NOISE = IdleNoise(x_rate=0.001, y_rate=0.001, z_rate=0.001)


@dataclass(frozen=True)
class MemoryShot:
    """One shot: every measured syndrome followed by the final data readout."""

    syndromes: tuple[Syndrome, ...]
    data_bits: DataBits

    def __post_init__(self) -> None:
        for syndrome in self.syndromes:
            validate_binary_bits("each syndrome", syndrome, len(PATCH.ancillas))
        validate_binary_bits("data readout", self.data_bits, len(PATCH.data_qubits))

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


def idle_duration_per_round(total_time_us: float, rounds: int, round_duration_us: float) -> float:
    """Return free-evolution time in each slice of a fixed storage window."""
    validate_nonnegative_number("total_time_us", total_time_us)
    validate_nonnegative_number("round_duration_us", round_duration_us)
    _validate_rounds(rounds)
    if rounds == 0:
        return total_time_us
    interval = total_time_us / rounds
    if round_duration_us > interval:
        raise ValueError(
            f"{rounds} rounds of {round_duration_us:g} us do not fit in {total_time_us:g} us"
        )
    return interval - round_duration_us


def _build_memory_circuit(
    rounds: int,
    error: Pauli | None,
    *,
    basis: str,
    idle_duration_us: float | None,
    preparation: str,
) -> QuantumCircuit:
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

    _validate_rounds(rounds)
    _validate_preparation(preparation)
    checked_basis = normalize_measurement_basis(basis)
    qubits = QuantumRegister(NUM_QUBITS, "q")
    syndrome_registers = [
        ClassicalRegister(len(PATCH.ancillas), f"syn_{round_index}")
        for round_index in range(rounds)
    ]
    data = ClassicalRegister(len(PATCH.data_qubits), "data")
    circuit = QuantumCircuit(qubits, *syndrome_registers, data)
    if preparation == "encoder":
        circuit.compose(_logical_state_circuit(checked_basis), PATCH.data_qubits, inplace=True)
    elif checked_basis == "X":
        circuit.h(PATCH.data_qubits)
    if error is not None:
        _apply_error(circuit, error)

    for syndrome_register in syndrome_registers:
        if idle_duration_us:
            circuit.delay(idle_duration_us, PATCH.data_qubits, unit="us")
        circuit.reset(PATCH.ancillas)
        _append_syndrome_round(circuit, syndrome_register)
        circuit.barrier()

    if not rounds and idle_duration_us:
        circuit.delay(idle_duration_us, PATCH.data_qubits, unit="us")

    if checked_basis == "X":
        circuit.h(PATCH.data_qubits)
    for index, qubit in enumerate(PATCH.data_qubits):
        circuit.measure(qubit, data[index])
    return circuit


def to_memory_circuit(
    rounds: int,
    error: Pauli | None = None,
    *,
    basis: str = "Z",
    preparation: str = "encoder",
) -> QuantumCircuit:
    """Prepare a logical +1 state, extract syndromes, then measure its basis."""
    return _build_memory_circuit(
        rounds,
        error,
        basis=basis,
        idle_duration_us=None,
        preparation=preparation,
    )


def to_timed_memory_circuit(
    total_time_us: float,
    rounds: int,
    *,
    round_duration_us: float,
    basis: str = "Z",
    error: Pauli | None = None,
    preparation: str = "encoder",
) -> QuantumCircuit:
    """Build a fixed-duration memory circuit with equally spaced check rounds."""
    idle_duration_us = idle_duration_per_round(total_time_us, rounds, round_duration_us)
    return _build_memory_circuit(
        rounds,
        error,
        basis=basis,
        idle_duration_us=idle_duration_us,
        preparation=preparation,
    )


def _noise_model(
    noise: CircuitNoise,
    *,
    idle_noise: IdleNoise | None = None,
    idle_duration_us: float = 0.0,
) -> NoiseModel:
    from qiskit_aer.noise import NoiseModel, ReadoutError, depolarizing_error, pauli_error

    model = NoiseModel()
    if noise.single_qubit:
        one_qubit_error = depolarizing_error(noise.single_qubit, 1)
        model.add_all_qubit_quantum_error(
            one_qubit_error,
            ("h", "s", "sdg", "x", "y", "z"),
        )
    if noise.two_qubit:
        model.add_all_qubit_quantum_error(
            depolarizing_error(noise.two_qubit, 2),
            ("cx", "swap"),
        )
    if noise.readout:
        readout_error = ReadoutError(
            [[1 - noise.readout, noise.readout], [noise.readout, 1 - noise.readout]]
        )
        model.add_all_qubit_readout_error(readout_error)
    if noise.reset:
        reset_error = pauli_error((("X", noise.reset), ("I", 1 - noise.reset)))
        model.add_all_qubit_quantum_error(reset_error, ("reset",))
    if idle_noise is not None and idle_duration_us:
        probability_i, probability_x, probability_y, probability_z = idle_noise.pauli_probabilities(
            idle_duration_us
        )
        idle_error = pauli_error(
            (
                ("I", probability_i),
                ("X", probability_x),
                ("Y", probability_y),
                ("Z", probability_z),
            )
        )
        model.add_all_qubit_quantum_error(idle_error, ("delay",))
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
    basis: str = "Z",
    preparation: str = "encoder",
) -> dict[MemoryShot, int]:
    """Run a repeated-round memory circuit and tally complete shot histories."""
    _validate_rounds(rounds)
    _validate_aer_options(shots, seed)
    noise = BASELINE_NOISE if noise is None else noise
    if not isinstance(noise, CircuitNoise):
        raise TypeError(f"noise must be CircuitNoise, got {type(noise).__name__}")

    circuit = to_memory_circuit(rounds, error, basis=basis, preparation=preparation)
    model = None if noise.is_ideal else _noise_model(noise)
    return _execute(circuit, model, rounds=rounds, shots=shots, seed=seed)


def run_timed_memory(
    total_time_us: float,
    rounds: int,
    *,
    round_duration_us: float,
    shots: int = 1024,
    seed: int | None = None,
    error: Pauli | None = None,
    noise: CircuitNoise | None = None,
    idle_noise: IdleNoise | None = None,
    basis: str = "Z",
    preparation: str = "encoder",
) -> dict[MemoryShot, int]:
    """Run a memory experiment whose storage window is fixed in microseconds."""
    _validate_aer_options(shots, seed)
    checked_basis = normalize_measurement_basis(basis)
    circuit_noise = BASELINE_NOISE if noise is None else noise
    storage_noise = BASELINE_IDLE_NOISE if idle_noise is None else idle_noise
    if not isinstance(circuit_noise, CircuitNoise):
        raise TypeError(f"noise must be CircuitNoise, got {type(circuit_noise).__name__}")
    if not isinstance(storage_noise, IdleNoise):
        raise TypeError(f"idle_noise must be IdleNoise, got {type(storage_noise).__name__}")

    idle_duration_us = idle_duration_per_round(total_time_us, rounds, round_duration_us)
    circuit = _build_memory_circuit(
        rounds,
        error,
        basis=checked_basis,
        idle_duration_us=idle_duration_us,
        preparation=preparation,
    )

    model = (
        None
        if circuit_noise.is_ideal and storage_noise.is_ideal
        else _noise_model(
            circuit_noise, idle_noise=storage_noise, idle_duration_us=idle_duration_us
        )
    )
    return _execute(circuit, model, rounds=rounds, shots=shots, seed=seed)


def _execute(
    circuit: QuantumCircuit,
    noise_model: NoiseModel | None,
    *,
    rounds: int,
    shots: int,
    seed: int | None,
) -> dict[MemoryShot, int]:
    """Run one memory circuit on the Aer stabilizer backend and tally shot histories."""
    from qiskit_aer import AerSimulator

    backend_options: dict[str, object] = {"method": "stabilizer"}
    if noise_model is not None:
        backend_options["noise_model"] = noise_model
    run_options: dict[str, int] = {"shots": shots}
    if seed is not None:
        run_options["seed_simulator"] = seed
    result = AerSimulator(**backend_options).run(circuit, **run_options).result()

    tallies: dict[MemoryShot, int] = {}
    for key, count in result.get_counts().items():
        parsed = _parse_memory_shot(key, rounds)
        tallies[parsed] = tallies.get(parsed, 0) + count
    return tallies


def summarize_memory(tallies: dict[MemoryShot, int], *, basis: str = "Z") -> MemorySummary:
    """Summarize raw Z-basis histories without applying the syndrome-history decoder.

    The tallies do not record their measurement basis, so the caller states it;
    only the Z basis is summarized here. Use ``run_cadence_experiment`` for a
    decoded X- or Z-basis result.
    """
    if normalize_measurement_basis(basis) != "Z":
        raise ValueError("summarize_memory only supports the Z basis; use run_cadence_experiment")
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
            last_round_successes += count * shot_z_success(shot.syndromes[-1], shot.data_bits)

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
