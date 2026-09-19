"""Aer runner for abstract heavy-hex memory circuits.

Parses counts into per-shot gauge histories, derives checks per round, and
grades each shot by subsystem lookup decoding: correct the final data bits
with the min-weight correction, then test the logical parity.
"""

from __future__ import annotations

from .._validation import validate_binary_bits
from ..circuits.heavyhex import MemorySchedule, memory_circuit
from ..circuits.heavyhex_flagged import FlaggedSchedule, memory_circuit_flagged
from ..core import Pauli
from ..decoders.heavyhex_lookup import decode
from ..patches.heavyhex import D3, HeavyHexOperators

MAX_SEED = (1 << 63) - 1


def _bits_le(bitstring: str) -> tuple[int, ...]:
    return tuple(int(bit) for bit in bitstring[::-1])


def _validate_options(shots: int, seed: int | None) -> None:
    if not isinstance(shots, int) or isinstance(shots, bool) or shots <= 0:
        raise ValueError(f"shots must be a positive integer, got {shots!r}")
    if seed is not None and (
        not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 or seed > MAX_SEED
    ):
        raise ValueError(f"seed must be an integer from 0 to {MAX_SEED}, or None, got {seed!r}")


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
) -> list[dict]:
    """Return one record per shot: gauge bits, data bits, checks, success."""
    _validate_options(shots, seed)
    from qiskit_aer import AerSimulator

    circuit, schedule = memory_circuit(
        patch, rounds=rounds, basis=basis, error=error, inject_at=inject_at
    )
    run_options: dict[str, int] = {"shots": shots}
    if seed is not None:
        run_options["seed_simulator"] = seed
    counts = AerSimulator(method="stabilizer").run(circuit, **run_options).result().get_counts()

    n = patch.distance * patch.distance
    n_gauge = len(schedule.gauges)
    logical = patch.code.logical_x if basis == "X" else patch.code.logical_z
    logical_support = set(logical.x or logical.z)
    grade = inject_at == "after_prep" and rounds == 1
    records: list[dict] = []
    for key, count in counts.items():
        data_str, gauge_str = key.split()
        data_bits, gauge_bits = _bits_le(data_str), _bits_le(gauge_str)
        validate_binary_bits("data readout", data_bits, n)
        validate_binary_bits("gauge outcomes", gauge_bits, n_gauge)
        checks = schedule.checks(gauge_bits)
        syndrome = syndrome_from_checks(schedule, checks)
        success: bool | None = None
        if grade:
            correction = decode(syndrome, patch.code)
            bits = list(data_bits)
            flip = correction.x if basis == "Z" else correction.z
            for qubit in flip:
                bits[qubit] ^= 1
            success = sum(bits[q] for q in logical_support) % 2 == 0
        records.extend(
            [
                {
                    "gauge_bits": gauge_bits,
                    "data_bits": data_bits,
                    "checks": checks,
                    "syndrome": syndrome,
                    "success": success,
                }
            ]
            * count
        )
    return records


def _deflag_adjusted_x_checks(
    schedule: FlaggedSchedule,
    checks: dict[tuple[int, str], int],
    corrections: list[tuple[tuple[int, str], Pauli]],
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


def flagged_syndrome(
    schedule: FlaggedSchedule,
    gauge_bits: tuple[int, ...],
) -> tuple[tuple[int, ...], list[tuple[tuple[int, str], Pauli]]]:
    """Round-0 detectors for an error injected right after prep.

    X checks are deflag-adjusted, then differenced against the prep half when
    the prep measured X; Z checks likewise without adjustment (Z deflag
    commutes with Z checks). Returns (syndrome, deflag corrections).
    """
    corrections = schedule.deflag_corrections(gauge_bits) + schedule.backaction_residuals(
        gauge_bits
    )
    checks = _deflag_adjusted_x_checks(schedule, schedule.checks(gauge_bits), corrections)
    bits = []
    for name in schedule.patch.stabilizer_names:
        bit = checks[0, name]
        if (-1, name) in checks:
            bit ^= checks[-1, name]
        bits.append(bit)
    return tuple(bits), corrections


def run_flagged_circuit(
    circuit,
    schedule: FlaggedSchedule,
    *,
    shots: int = 1024,
    seed: int | None = None,
    grade: bool = True,
) -> list[dict]:
    """Parse Aer counts for a (possibly fault-injected) flagged circuit."""
    _validate_options(shots, seed)
    from qiskit_aer import AerSimulator

    run_options: dict[str, int] = {"shots": shots}
    if seed is not None:
        run_options["seed_simulator"] = seed
    counts = AerSimulator(method="stabilizer").run(circuit, **run_options).result().get_counts()

    n = schedule.patch.distance * schedule.patch.distance
    n_gauge = len(schedule.measurements)
    basis = schedule.basis
    logical = schedule.patch.code.logical_x if basis == "X" else schedule.patch.code.logical_z
    logical_support = set(logical.x or logical.z)
    grade &= schedule.rounds == 1
    records: list[dict] = []
    for key, count in counts.items():
        data_str, gauge_str = key.split()
        data_bits, gauge_bits = _bits_le(data_str), _bits_le(gauge_str)
        validate_binary_bits("data readout", data_bits, n)
        validate_binary_bits("gauge outcomes", gauge_bits, n_gauge)
        checks = schedule.checks(gauge_bits)
        syndrome, corrections = flagged_syndrome(schedule, gauge_bits)
        success: bool | None = None
        if grade:
            correction = decode(syndrome, schedule.patch.code)
            bits = list(data_bits)
            if basis == "X":
                for _, deflag in corrections:
                    for qubit in deflag.z:
                        bits[qubit] ^= 1
            flip = correction.x if basis == "Z" else correction.z
            for qubit in flip:
                bits[qubit] ^= 1
            success = sum(bits[q] for q in logical_support) % 2 == 0
        records.extend(
            [
                {
                    "gauge_bits": gauge_bits,
                    "data_bits": data_bits,
                    "checks": checks,
                    "syndrome": syndrome,
                    "corrections": corrections,
                    "success": success,
                }
            ]
            * count
        )
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
) -> list[dict]:
    """Flagged memory experiment: same records as run_flagged_circuit."""
    circuit, schedule = memory_circuit_flagged(
        patch, rounds=rounds, basis=basis, error=error, inject_at=inject_at
    )
    return run_flagged_circuit(
        circuit, schedule, shots=shots, seed=seed, grade=inject_at == "after_prep"
    )
