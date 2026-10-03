"""Sample the flagged circuit on Aer's stabilizer simulator, a check independent of Stim."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .._validation import validate_binary_bits, validate_shots_and_seed
from ..circuits.flagged import FlaggedSchedule, memory_circuit_flagged
from ..circuits.shots import read_shot
from ..core import Pauli
from ..patches.operators import D3, HeavyHexOperators

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

Record = dict[str, Any]


def _sample_counts(circuit: QuantumCircuit, shots: int, seed: int | None) -> dict[str, int]:
    from qiskit_aer import AerSimulator  # optional extra

    options: dict[str, int] = {"shots": shots}
    if seed is not None:
        options["seed_simulator"] = seed
    return AerSimulator(method="stabilizer").run(circuit, **options).result().get_counts()


def _split_key(key: str, n_data: int, n_gauge: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Aer count key -> (data bits, gauge bits), each with bit 0 first."""
    data_str, gauge_str = key.split()
    data_bits = tuple(int(bit) for bit in data_str[::-1])
    gauge_bits = tuple(int(bit) for bit in gauge_str[::-1])
    validate_binary_bits("data readout", data_bits, n_data)
    validate_binary_bits("gauge outcomes", gauge_bits, n_gauge)
    return data_bits, gauge_bits


def run_flagged_circuit(
    circuit: QuantumCircuit,
    schedule: FlaggedSchedule,
    *,
    shots: int = 1024,
    seed: int | None = None,
) -> list[Record]:
    """Sample a flagged circuit; one record per shot with its syndrome, detectors and logical bit."""
    validate_shots_and_seed(shots, seed)
    code = schedule.patch.code
    records: list[Record] = []
    for key, count in _sample_counts(circuit, shots, seed).items():
        data_bits, gauge_bits = _split_key(key, code.n, len(schedule.measurements))
        shot = read_shot(schedule, gauge_bits, data_bits)
        record: Record = {
            "gauge_bits": gauge_bits,
            "data_bits": data_bits,
            "syndrome": shot.detectors[: code.syndrome_size],
            "detectors": shot.detectors,
            "logical_bit": shot.logical_bit,
        }
        records.extend([record] * count)  # identical shots share one record
    return records


def run_memory_flagged(
    patch: HeavyHexOperators = D3,
    *,
    rounds: int = 1,
    basis: str = "Z",
    error: Pauli | None = None,
    inject_at: str = "after_prep",
    shots: int = 1024,
    seed: int | None = None,
) -> list[Record]:
    """Build the flagged circuit and sample it; same records as run_flagged_circuit."""
    circuit, schedule = memory_circuit_flagged(
        patch, rounds=rounds, basis=basis, error=error, inject_at=inject_at
    )
    return run_flagged_circuit(circuit, schedule, shots=shots, seed=seed)
