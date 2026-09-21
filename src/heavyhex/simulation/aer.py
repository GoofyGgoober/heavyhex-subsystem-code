"""Aer runners for the heavy-hex memory experiments, graded by lookup decoding."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

from .._validation import validate_binary_bits
from ..circuits.flagged import FlaggedSchedule, memory_circuit_flagged
from ..circuits.ideal import MemorySchedule, memory_circuit
from ..core import Pauli
from ..decoders.lookup import decode
from ..patches.operators import D3, HeavyHexOperators

if TYPE_CHECKING:
    from qiskit import QuantumCircuit

MAX_SEED = (1 << 63) - 1

Record = dict[str, Any]
# Virtual Pauli corrections tagged by the (round, half) they apply after.
Corrections = list[tuple[tuple[int, str], Pauli]]


def _validate_options(shots: int, seed: int | None) -> None:
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots!r}")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed > MAX_SEED
    ):
        raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}, or None, got {seed!r}")


def _sample_counts(circuit: QuantumCircuit, shots: int, seed: int | None) -> dict[str, int]:
    from qiskit_aer import AerSimulator  # optional extra

    options: dict[str, int] = {"shots": shots}
    if seed is not None:
        options["seed_simulator"] = seed
    return AerSimulator(method="stabilizer").run(circuit, **options).result().get_counts()


def _bits_le(bitstring: str) -> tuple[int, ...]:
    return tuple(int(bit) for bit in bitstring[::-1])


def _shot_bits(key: str, n_data: int, n_gauge: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Split one Aer count key into little-endian (data bits, gauge bits)."""
    data_str, gauge_str = key.split()
    data_bits, gauge_bits = _bits_le(data_str), _bits_le(gauge_str)
    validate_binary_bits("data readout", data_bits, n_data)
    validate_binary_bits("gauge outcomes", gauge_bits, n_gauge)
    return data_bits, gauge_bits


def _logical_support(patch: HeavyHexOperators, basis: str) -> frozenset[int]:
    logical = patch.code.logical_x if basis == "X" else patch.code.logical_z
    return logical.x or logical.z


def _survives(data_bits: tuple[int, ...], flips: Iterable[int], support: frozenset[int]) -> bool:
    """True when the corrected readout still carries even logical parity."""
    bits = list(data_bits)
    for qubit in flips:
        bits[qubit] ^= 1
    return sum(bits[qubit] for qubit in support) % 2 == 0


def syndrome_from_checks(
    schedule: MemorySchedule, checks: dict[tuple[int, str], int]
) -> tuple[int, ...]:
    """Round-0 detectors for an error injected right after prep.

    Checks measured in the prep half are differenced against round 0; checks
    of the other basis start at +1 from the product-state prep.
    """
    prep_half = "X" if schedule.basis == "Z" else "Z"
    bits = []
    for name in schedule.patch.stabilizer_names:
        half = "X" if name.startswith("X") else "Z"
        bit = checks[0, name]
        if half == prep_half:
            bit ^= checks[-1, name]
        bits.append(bit)
    return tuple(bits)


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
    """One record per shot: gauge bits, data bits, checks, syndrome, success."""
    _validate_options(shots, seed)
    circuit, schedule = memory_circuit(
        patch, rounds=rounds, basis=basis, error=error, inject_at=inject_at
    )
    counts = _sample_counts(circuit, shots, seed)

    n_data = patch.distance * patch.distance
    support = _logical_support(patch, basis)
    grade = inject_at == "after_prep" and rounds == 1
    records: list[Record] = []
    for key, count in counts.items():
        data_bits, gauge_bits = _shot_bits(key, n_data, len(schedule.gauges))
        checks = schedule.checks(gauge_bits)
        syndrome = syndrome_from_checks(schedule, checks)
        success: bool | None = None
        if grade:
            correction = decode(syndrome, patch.code)
            success = _survives(data_bits, correction.x if basis == "Z" else correction.z, support)
        record: Record = {
            "gauge_bits": gauge_bits,
            "data_bits": data_bits,
            "checks": checks,
            "syndrome": syndrome,
            "success": success,
        }
        records.extend([record] * count)  # identical shots share one record
    return records


def _virtual_corrections(schedule: FlaggedSchedule, gauge_bits: tuple[int, ...]) -> Corrections:
    return schedule.deflag_corrections(gauge_bits) + schedule.backaction_residuals(gauge_bits)


def _deflag_adjusted_x_checks(
    schedule: FlaggedSchedule,
    checks: dict[tuple[int, str], int],
    corrections: Corrections,
) -> dict[tuple[int, str], int]:
    """Flip X checks measured after each virtual Z correction's Z half."""
    position = {half: index for index, half in enumerate(schedule.half_order)}
    stabs = {
        name: pauli
        for name, pauli in zip(schedule.patch.stabilizer_names, schedule.patch.code.stabilizers)
        if name.startswith("X")
    }
    adjusted = dict(checks)
    for (round, name), bit in checks.items():
        if not name.startswith("X"):
            continue
        flip = 0
        for correction_half, correction in corrections:
            if position[correction_half] < position[round, "X"] and not correction.commutes(
                stabs[name]
            ):
                flip ^= 1
        adjusted[round, name] = bit ^ flip
    return adjusted


def _syndrome_from_flagged_checks(
    schedule: FlaggedSchedule,
    checks: dict[tuple[int, str], int],
    corrections: Corrections,
) -> tuple[int, ...]:
    adjusted = _deflag_adjusted_x_checks(schedule, checks, corrections)
    bits = []
    for name in schedule.patch.stabilizer_names:
        bit = adjusted[0, name]
        if (-1, name) in adjusted:
            bit ^= adjusted[-1, name]
        bits.append(bit)
    return tuple(bits)


def flagged_syndrome(
    schedule: FlaggedSchedule, gauge_bits: tuple[int, ...]
) -> tuple[tuple[int, ...], Corrections]:
    """Round-0 detectors plus the deflag corrections they assume.

    X checks are deflag-adjusted before being differenced against the prep
    half; Z deflag corrections commute with the Z checks, so those pass through.
    """
    corrections = _virtual_corrections(schedule, gauge_bits)
    checks = schedule.checks(gauge_bits)
    return _syndrome_from_flagged_checks(schedule, checks, corrections), corrections


def run_flagged_circuit(
    circuit: QuantumCircuit,
    schedule: FlaggedSchedule,
    *,
    shots: int = 1024,
    seed: int | None = None,
    grade: bool = True,
) -> list[Record]:
    """Parse Aer counts for a (possibly fault-injected) flagged circuit."""
    _validate_options(shots, seed)
    counts = _sample_counts(circuit, shots, seed)

    patch = schedule.patch
    basis = schedule.basis
    n_data = patch.distance * patch.distance
    support = _logical_support(patch, basis)
    grade = grade and schedule.rounds == 1
    records: list[Record] = []
    for key, count in counts.items():
        data_bits, gauge_bits = _shot_bits(key, n_data, len(schedule.measurements))
        checks = schedule.checks(gauge_bits)
        corrections = _virtual_corrections(schedule, gauge_bits)
        syndrome = _syndrome_from_flagged_checks(schedule, checks, corrections)
        success: bool | None = None
        if grade:
            correction = decode(syndrome, patch.code)
            flips = list(correction.x if basis == "Z" else correction.z)
            if basis == "X":
                # X readout sees the virtual Z corrections; Z readout does not.
                flips += [qubit for _, deflag in corrections for qubit in deflag.z]
            success = _survives(data_bits, flips, support)
        record: Record = {
            "gauge_bits": gauge_bits,
            "data_bits": data_bits,
            "checks": checks,
            "syndrome": syndrome,
            "corrections": corrections,
            "success": success,
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
    """Flagged memory experiment; records match run_flagged_circuit."""
    circuit, schedule = memory_circuit_flagged(
        patch, rounds=rounds, basis=basis, error=error, inject_at=inject_at
    )
    return run_flagged_circuit(
        circuit, schedule, shots=shots, seed=seed, grade=inject_at == "after_prep"
    )
