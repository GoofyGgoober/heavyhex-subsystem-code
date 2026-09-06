"""Fixed-duration syndrome-cadence experiment and statistical summaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist

from ..decoders.infer_logical import (
    logical_bit_from_data_readout,
    logical_readout_success,
    normalize_measurement_basis,
    select_check_results,
)
from ..patches import PATCH
from .aer import MAX_SEED
from .record_shots import (
    BASELINE_IDLE_NOISE,
    BASELINE_NOISE,
    CircuitNoise,
    IdleNoise,
    idle_duration_per_round,
    run_timed_memory,
)


def _odd_parity_probability(probabilities: Sequence[float]) -> float:
    parity_bias = 1.0
    for probability in probabilities:
        parity_bias *= 1 - 2 * probability
    return (1 - parity_bias) / 2


def _round_data_probability(noise: CircuitNoise) -> float:
    """Approximate one-round data-fault weight used only by the decoder."""
    marginal_cx_flip = noise.two_qubit / 2
    per_qubit = []
    for qubit in PATCH.data_qubits:
        degree = sum(qubit in check.support for check in PATCH.checks)
        per_qubit.append(_odd_parity_probability((marginal_cx_flip,) * degree))
    return sum(per_qubit) / len(per_qubit)


def _syndrome_bit_probability(noise: CircuitNoise, basis: str) -> float:
    """Approximate syndrome-fault weight used only by the decoder."""
    checked_basis = normalize_measurement_basis(basis)
    probabilities = []
    for check in PATCH.checks:
        if check.basis != checked_basis:
            continue
        opportunities = [noise.readout, noise.reset]
        opportunities.extend((noise.two_qubit / 2,) * len(check.support))
        if checked_basis == "X":
            opportunities.extend((noise.single_qubit / 2,) * 2)
        probabilities.append(_odd_parity_probability(opportunities))
    return sum(probabilities) / len(probabilities)


def _terminal_probability(noise: CircuitNoise, basis: str) -> float:
    checked_basis = normalize_measurement_basis(basis)
    opportunities = [noise.readout]
    if checked_basis == "X":
        opportunities.append(noise.single_qubit / 2)
    return _odd_parity_probability(opportunities)


def _wilson_interval(failures: int, shots: int, confidence: float) -> tuple[float, float]:
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence!r}")
    z_score = NormalDist().inv_cdf((1 + confidence) / 2)
    estimate = failures / shots
    denominator = 1 + z_score**2 / shots
    center = (estimate + z_score**2 / (2 * shots)) / denominator
    radius = (
        z_score
        * (
            estimate * (1 - estimate) / shots
            + z_score**2 / (4 * shots**2)
        )
        ** 0.5
        / denominator
    )
    return center - radius, center + radius


@dataclass(frozen=True)
class CadencePoint:
    """Measured logical performance for one number of syndrome rounds."""

    rounds: int
    interval_us: float
    idle_per_round_us: float
    shots: int
    logical_failures: int
    logical_failure_rate: float
    confidence_low: float
    confidence_high: float
    raw_logical_failure_rate: float
    mean_detection_events: float
    decoder_interval_bit_error: float
    decoder_syndrome_bit_error: float
    decoder_terminal_bit_error: float


@dataclass(frozen=True)
class CadenceExperiment:
    """A fixed-duration cadence sweep in one logical measurement basis."""

    basis: str
    total_time_us: float
    round_duration_us: float
    confidence: float
    bare_qubit_failure_rate: float
    points: tuple[CadencePoint, ...]
    preparation: str = "encoder"

    @property
    def optimum_rounds(self) -> int:
        return min(
            self.points,
            key=lambda point: (point.logical_failure_rate, point.rounds),
        ).rounds


def _decoder_probabilities(
    *,
    basis: str,
    rounds: int,
    idle_per_round_us: float,
    circuit_noise: CircuitNoise,
    idle_noise: IdleNoise,
) -> tuple[float, float, float]:
    idle_bit_error = idle_noise.relevant_bit_probability(basis, idle_per_round_us)
    terminal = _terminal_probability(circuit_noise, basis)
    if rounds == 0:
        return 0.0, 0.0, _odd_parity_probability((idle_bit_error, terminal))
    interval = _odd_parity_probability(
        (idle_bit_error, _round_data_probability(circuit_noise))
    )
    syndrome = _syndrome_bit_probability(circuit_noise, basis)
    return interval, syndrome, terminal


def bare_qubit_failure_rate(
    total_time_us: float,
    *,
    basis: str,
    circuit_noise: CircuitNoise,
    idle_noise: IdleNoise,
) -> float:
    """Analytic no-check reference, excluding state-preparation faults."""
    idle = idle_noise.relevant_bit_probability(basis, total_time_us)
    terminal = _terminal_probability(circuit_noise, basis)
    return _odd_parity_probability((idle, terminal))


def run_cadence_experiment(
    total_time_us: float,
    rounds: Sequence[int],
    *,
    round_duration_us: float,
    basis: str,
    shots: int = 4096,
    seed: int | None = None,
    confidence: float = 0.95,
    circuit_noise: CircuitNoise | None = None,
    idle_noise: IdleNoise | None = None,
    preparation: str = "encoder",
) -> CadenceExperiment:
    """Measure logical failure as syndrome cadence varies at fixed total time.

    ``mean_detection_events`` counts syndrome changes on the checks the decoder
    reads (Z checks for a Z-basis test), so it is comparable across preparations.
    """
    checked_basis = normalize_measurement_basis(basis)
    selected_rounds = tuple(sorted(set(rounds)))
    if not selected_rounds:
        raise ValueError("rounds must contain at least one value")
    if any(
        not isinstance(value, int) or isinstance(value, bool) or value < 0
        for value in selected_rounds
    ):
        raise ValueError(f"rounds must contain non-negative integers, got {rounds!r}")
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots!r}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be between 0 and 1, got {confidence!r}")

    gate_noise = BASELINE_NOISE if circuit_noise is None else circuit_noise
    storage_noise = BASELINE_IDLE_NOISE if idle_noise is None else idle_noise
    points: list[CadencePoint] = []

    for point_index, round_count in enumerate(selected_rounds):
        idle_us = idle_duration_per_round(
            total_time_us,
            round_count,
            round_duration_us,
        )
        point_seed = None if seed is None else (seed + point_index) % (MAX_SEED + 1)
        tallies = run_timed_memory(
            total_time_us,
            round_count,
            round_duration_us=round_duration_us,
            shots=shots,
            seed=point_seed,
            noise=gate_noise,
            idle_noise=storage_noise,
            basis=checked_basis,
            preparation=preparation,
        )
        interval_error, syndrome_error, terminal_error = _decoder_probabilities(
            basis=checked_basis,
            rounds=round_count,
            idle_per_round_us=idle_us,
            circuit_noise=gate_noise,
            idle_noise=storage_noise,
        )
        logical_failures = 0
        raw_failures = 0
        detection_events = 0
        observed_shots = 0
        for shot, count in tallies.items():
            observed_shots += count
            raw_failures += count * logical_bit_from_data_readout(
                shot.data_bits, checked_basis
            )
            success = logical_readout_success(
                shot.syndromes,
                shot.data_bits,
                basis=checked_basis,
                interval_bit_error=interval_error,
                syndrome_bit_error=syndrome_error,
                terminal_bit_error=terminal_error,
            )
            logical_failures += count * (1 - success)
            detection_events += count * sum(
                sum(select_check_results(round_events, check_type=checked_basis))
                for round_events in shot.detection_events()
            )
        if observed_shots != shots:
            raise ValueError(f"expected {shots} outcomes, received {observed_shots}")
        confidence_low, confidence_high = _wilson_interval(
            logical_failures, shots, confidence
        )
        interval_us = total_time_us if round_count == 0 else total_time_us / round_count
        points.append(
            CadencePoint(
                rounds=round_count,
                interval_us=interval_us,
                idle_per_round_us=idle_us,
                shots=shots,
                logical_failures=logical_failures,
                logical_failure_rate=logical_failures / shots,
                confidence_low=confidence_low,
                confidence_high=confidence_high,
                raw_logical_failure_rate=raw_failures / shots,
                mean_detection_events=detection_events / shots,
                decoder_interval_bit_error=interval_error,
                decoder_syndrome_bit_error=syndrome_error,
                decoder_terminal_bit_error=terminal_error,
            )
        )

    return CadenceExperiment(
        basis=checked_basis,
        total_time_us=total_time_us,
        round_duration_us=round_duration_us,
        confidence=confidence,
        bare_qubit_failure_rate=bare_qubit_failure_rate(
            total_time_us,
            basis=checked_basis,
            circuit_noise=gate_noise,
            idle_noise=storage_noise,
        ),
        points=tuple(points),
        preparation=preparation,
    )


def run_full_cadence_experiment(
    total_time_us: float,
    rounds: Sequence[int],
    *,
    round_duration_us: float,
    shots: int = 4096,
    seed: int | None = None,
    confidence: float = 0.95,
    circuit_noise: CircuitNoise | None = None,
    idle_noise: IdleNoise | None = None,
    preparation: str = "encoder",
) -> tuple[CadenceExperiment, CadenceExperiment]:
    """Run both logical-Z and logical-X memory benchmarks."""
    z_experiment = run_cadence_experiment(
        total_time_us,
        rounds,
        round_duration_us=round_duration_us,
        basis="Z",
        shots=shots,
        seed=seed,
        confidence=confidence,
        circuit_noise=circuit_noise,
        idle_noise=idle_noise,
        preparation=preparation,
    )
    x_seed = None if seed is None else (seed + len(set(rounds))) % (MAX_SEED + 1)
    x_experiment = run_cadence_experiment(
        total_time_us,
        rounds,
        round_duration_us=round_duration_us,
        basis="X",
        shots=shots,
        seed=x_seed,
        confidence=confidence,
        circuit_noise=circuit_noise,
        idle_noise=idle_noise,
        preparation=preparation,
    )
    return z_experiment, x_experiment
