"""The flagged memory circuits on Fez, simulated in stim with noise from the calibration.

Each CZ gets its coupler's error, each measurement and reset its qubit's readout
error, and each qubit decays (T1, T2) while idle. MWPM matches on the circuit's
error model, so flag and hook errors are included.
"""

from __future__ import annotations

import statistics
import warnings
from dataclasses import dataclass, replace
from math import exp, sqrt
from typing import TYPE_CHECKING

from ..circuits.flagged import FlaggedSchedule, memory_circuit_flagged
from ..patches.operators import build_operators
from ..patches.placement import Calibration, broken_coupler, broken_qubit, fez_qubits

if TYPE_CHECKING:
    import stim

# Fez durations in ns. A CX runs as a CZ with an H on each side of the target.
CZ_NS, H_NS, MEASURE_NS, RESET_NS = 84, 24, 1560, 1584
H_ERROR = 2.3e-4  # Fez's median single-qubit gate error; the calibration file doesn't keep it


def typical(calibration: Calibration) -> Calibration:
    """Every coupler and qubit at the chip's median, nothing broken."""
    working = [q for q in calibration.readout if not broken_qubit(q, calibration)]
    cz = statistics.median(e for e in calibration.cz.values() if not broken_coupler(e))
    readout, t1, t2 = (
        statistics.median(values[q] for q in working)
        for values in (calibration.readout, calibration.t1_us, calibration.t2_us)
    )
    return replace(
        calibration,
        cz=dict.fromkeys(calibration.cz, cz),
        readout=dict.fromkeys(calibration.readout, readout),
        t1_us=dict.fromkeys(calibration.t1_us, t1),
        t2_us=dict.fromkeys(calibration.t2_us, t2),
        broken_qubits=(),
    )


def repaired(calibration: Calibration) -> Calibration:
    """Broken couplers and qubits set to the chip's median, as on a day they work."""
    median = typical(calibration)
    bad = {q for q in calibration.readout if broken_qubit(q, calibration)}

    def fix(values: dict, medians: dict) -> dict:
        return {q: medians[q] if q in bad else v for q, v in values.items()}

    return replace(
        calibration,
        cz={c: median.cz[c] if broken_coupler(e) else e for c, e in calibration.cz.items()},
        readout=fix(calibration.readout, median.readout),
        t1_us=fix(calibration.t1_us, median.t1_us),
        t2_us=fix(calibration.t2_us, median.t2_us),
        broken_qubits=(),
    )


def noisy_circuit(
    distance: int,
    basis: str,
    rounds: int,
    calibration: Calibration,
    *,
    scale: float = 1.0,
    decoupling: bool = False,
) -> stim.Circuit:
    """The flagged memory circuit on its Fez qubits, with noise, detectors and the logical.

    scale multiplies every error probability (for idling, the idle time).
    decoupling stands in for dynamical decoupling: T2 = 2 T1.
    """
    import stim

    circuit, schedule = memory_circuit_flagged(
        build_operators(distance), rounds=rounds, basis=basis
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".* uses weak parts")
        chip = fez_qubits(distance, calibration)

    out = stim.Circuit()
    free: dict[int, float] = {}  # when each qubit is next free, in ns
    just_reset: set[int] = set()  # not in a CX since its reset; on the chip the reset waits instead
    records: dict[tuple[str, int], int] = {}  # (register, bit) -> measurement index

    def scaled(p: float) -> float:
        return min(scale * p, 0.5)

    def wait(q: int, until: float) -> None:
        if q in free and q not in just_reset and until > free[q]:
            t = scale * (until - free[q])
            t1 = calibration.t1_us[chip[q]] * 1e3
            t2 = 2 * t1 if decoupling else calibration.t2_us[chip[q]] * 1e3
            px = (1 - exp(-t / t1)) / 4
            pz = max((1 - exp(-t / t2)) / 2 - px, 0)
            out.append("PAULI_CHANNEL_1", [q], [px, px, pz])
        free[q] = until

    for item in circuit.data:
        name = item.operation.name
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        start = max(free.get(q, 0) for q in qubits)
        for q in qubits:
            wait(q, start)
        if name == "barrier":
            out.append("TICK")
            continue
        if name == "h":
            out.append("H", qubits)
            out.append("DEPOLARIZE1", qubits, scaled(1.5 * H_ERROR))
            duration = H_NS
        elif name == "cx":
            control, target = qubits
            error = calibration.cz[tuple(sorted((chip[control], chip[target])))]
            out.append("CX", qubits)
            # IBM quotes average gate infidelity; a depolarizing channel needs 5/4 of it.
            out.append("DEPOLARIZE2", qubits, scaled(1.25 * error))
            out.append("DEPOLARIZE1", [target], scaled(3 * H_ERROR))
            just_reset.difference_update(qubits)
            duration = CZ_NS + 2 * H_NS
        elif name == "reset":
            (q,) = qubits
            out.append("R", [q])
            out.append("X_ERROR", [q], scaled(calibration.readout[chip[q]]))
            just_reset.add(q)
            duration = RESET_NS
        elif name == "measure":
            (q,) = qubits
            out.append("M", [q], scaled(calibration.readout[chip[q]]))
            bit = circuit.find_bit(item.clbits[0])
            register, index = bit.registers[0]
            records[register.name, index] = len(records)
            duration = MEASURE_NS
        else:
            raise ValueError(f"unexpected instruction: {name}")
        for q in qubits:
            free[q] = start + duration

    _add_detectors(out, schedule, records)
    return out


def logical_errors(circuit: stim.Circuit, shots: int, seed: int | None = None) -> int:
    """How many sampled shots MWPM decodes to the wrong logical value."""
    import numpy as np
    import pymatching

    model = circuit.detector_error_model(decompose_errors=True)
    matching = pymatching.Matching.from_detector_error_model(model)
    sampler = circuit.compile_detector_sampler(seed=seed)
    detectors, observables = sampler.sample(shots, separate_observables=True)
    predicted = matching.decode_batch(detectors)
    return int(np.count_nonzero(predicted[:, 0] != observables[:, 0]))


@dataclass(frozen=True)
class Decay:
    """Logical error after each number of rounds, and the fitted error per round."""

    rounds: tuple[int, ...]
    logical_error: tuple[float, ...]
    per_round: float
    uncertainty: float  # one standard error of per_round, from shot noise


def decay(
    distance: int,
    basis: str,
    calibration: Calibration,
    *,
    rounds: tuple[int, ...] = (1, 2, 3, 4, 6, 8),
    shots: int = 20_000,
    seed: int | None = None,
    scale: float = 1.0,
    decoupling: bool = False,
) -> Decay:
    """Fit log(1 - 2 p_L) = a + b n, each point weighted by its shot noise.

    The error per round is (1 - e^b) / 2; a takes up prep and readout error.
    Points with p_L = 0 or p_L >= 0.5 say nothing about b and are left out.
    """
    import numpy as np

    errors = []
    for n in rounds:
        circuit = noisy_circuit(distance, basis, n, calibration, scale=scale, decoupling=decoupling)
        errors.append(logical_errors(circuit, shots, None if seed is None else seed + n) / shots)
    p, n = np.array(errors), np.array(rounds, dtype=float)
    keep = (p > 0) & (p < 0.5)
    if keep.sum() < 2:
        return Decay(tuple(rounds), tuple(errors), 0.5, float("nan"))
    p, n = p[keep], n[keep]
    sigma = 2 * np.sqrt(p * (1 - p) / shots) / (1 - 2 * p)  # shot noise on log(1 - 2p)
    (b, _), cov = np.polyfit(n, np.log(1 - 2 * p), 1, w=1 / sigma, cov="unscaled")
    return Decay(tuple(rounds), tuple(errors), (1 - exp(b)) / 2, exp(b) / 2 * sqrt(cov[0, 0]))


def _add_detectors(
    out: stim.Circuit, schedule: FlaggedSchedule, records: dict[tuple[str, int], int]
) -> None:
    """Detectors on the memory-basis stabilizers, and the logical.

    A detector is a stabilizer's value XOR its value the round before. In Z
    memory every flag and relay is a detector too: each should read 0.
    """
    import stim

    patch, basis, rounds = schedule.patch, schedule.basis, schedule.rounds
    total = len(records)

    def rec(register: str, bit: int) -> stim.GateTarget:
        return stim.target_rec(records[register, bit] - total)

    bits = {(m.round, m.half, m.kind, m.gauge): m.bit for m in schedule.measurements}
    stabilizers = patch.x_stabilizers if basis == "X" else patch.z_stabilizers
    kind = "x_gauge" if basis == "X" else "z_syn"

    def outcomes(r: int, name: str) -> list[stim.GateTarget]:
        return [rec("m", bits[r, basis, kind, g]) for g in patch.stabilizer_gauge_factors[name]]

    for r in range(rounds):
        for name in stabilizers:
            previous = outcomes(r - 1, name) if r else []
            out.append("DETECTOR", outcomes(r, name) + previous)
    for name, labels in stabilizers.items():
        data = [rec("d", q - 1) for q in sorted(labels)]
        out.append("DETECTOR", data + outcomes(rounds - 1, name))
    if basis == "Z":
        for m in schedule.measurements:
            if m.kind in ("z_flag", "relay"):
                out.append("DETECTOR", [rec("m", m.bit)])
    logical = patch.code.logical_x if basis == "X" else patch.code.logical_z
    out.append("OBSERVABLE_INCLUDE", [rec("d", q) for q in sorted(logical.x or logical.z)], 0)
