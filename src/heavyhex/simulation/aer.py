"""Run memory circuits on Aer's stabilizer simulator and grade each shot."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from .._validation import validate_binary_bits, validate_shots_and_seed
from ..circuits.flagged import FlaggedSchedule, memory_circuit_flagged
from ..circuits.ideal import MemorySchedule, memory_circuit
from ..core import Pauli
from ..decoders.lookup import decode
from ..decoders.shots import Shot, read_shot
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


def _lookup_fails(shot: Shot) -> bool:
    """Decode the round-0 syndrome with the lookup table and check the logical readout."""
    code = shot.memory.patch.code
    return shot.logical_error(decode(shot.detectors[: code.syndrome_size], code))


def _records(
    schedule: MemorySchedule | FlaggedSchedule,
    counts: dict[str, int],
    fails: Callable[[Shot], bool] | None,
) -> list[Record]:
    """One record per shot, graded with `fails` when it is given."""
    code = schedule.patch.code
    records: list[Record] = []
    for key, count in counts.items():
        data_bits, gauge_bits = _split_key(key, code.n, len(schedule.measurements))
        shot = read_shot(schedule, gauge_bits, data_bits)
        record: Record = {
            "gauge_bits": gauge_bits,
            "data_bits": data_bits,
            "checks": schedule.checks(gauge_bits),
            "syndrome": shot.detectors[: code.syndrome_size],
            "detectors": shot.detectors,
            "success": None if fails is None else not fails(shot),
        }
        records.extend([record] * count)  # identical shots share one record
    return records


def run_memory(
    patch: HeavyHexOperators = D3,
    *,
    rounds: int = 1,
    basis: str = "Z",
    error: Pauli | None = None,
    inject_at: str = "after_prep",
    shots: int = 1024,
    seed: int | None = None,
) -> list[Record]:
    """Run the ideal circuit. Only one round with the error right after prep gets graded."""
    validate_shots_and_seed(shots, seed)
    circuit, schedule = memory_circuit(
        patch, rounds=rounds, basis=basis, error=error, inject_at=inject_at
    )
    grade = inject_at == "after_prep" and rounds == 1
    return _records(
        schedule, _sample_counts(circuit, shots, seed), _lookup_fails if grade else None
    )


def run_flagged_circuit(
    circuit: QuantumCircuit,
    schedule: FlaggedSchedule,
    *,
    shots: int = 1024,
    seed: int | None = None,
    grade: bool = True,
    decoder: str = "lookup",
) -> list[Record]:
    """Sample a flagged circuit and grade the shots.

    success is None for shots that aren't graded. Lookup can only grade d=3 with one round.
    """
    validate_shots_and_seed(shots, seed)
    if decoder not in ("lookup", "mwpm"):
        raise ValueError("decoder must be lookup or mwpm")
    fails = None
    if decoder == "mwpm":
        from ..decoders.mwpm import MWPMDecoder

        matcher = MWPMDecoder(schedule.patch, basis=schedule.basis, rounds=schedule.rounds)
        fails = matcher.fails if grade else None
    elif grade and schedule.rounds == 1 and schedule.patch.distance == 3:
        fails = _lookup_fails
    return _records(schedule, _sample_counts(circuit, shots, seed), fails)


def run_memory_flagged(
    patch: HeavyHexOperators = D3,
    *,
    rounds: int = 1,
    basis: str = "Z",
    error: Pauli | None = None,
    inject_at: str = "after_prep",
    shots: int = 1024,
    seed: int | None = None,
    decoder: str = "lookup",
) -> list[Record]:
    """Build the flagged circuit and run it; same records as run_flagged_circuit."""
    circuit, schedule = memory_circuit_flagged(
        patch, rounds=rounds, basis=basis, error=error, inject_at=inject_at
    )
    return run_flagged_circuit(
        circuit,
        schedule,
        shots=shots,
        seed=seed,
        grade=decoder == "mwpm" or inject_at == "after_prep",
        decoder=decoder,
    )
