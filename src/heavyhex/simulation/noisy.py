"""The flagged memory circuits on Fez, simulated in stim with noise from the calibration.

Each CZ gets its coupler's error, each measurement and reset its qubit's readout
error, and each qubit decays (T1, T2) while idle. With decoupling, long idles get
a pair of X pulses. MWPM matches on the circuit's error model, so flag and hook
errors are included.
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

# Fez's durations in ns, for calibrations saved without the device's own. H and X
# take one SX pulse; a CX runs as a CZ with an H on each side of the target.
FEZ_NS = {"x": 24, "cz": 84, "measure": 1560, "reset": 1584}
# Median X and SX error on Fez, Kingston and Marrakesh, for calibrations saved without each
# qubit's own.
SINGLE_QUBIT_ERROR = 2.3e-4
# Idles at least this long get pulses: the waits through readout and reset.
# Padding the gaps between gates as well changes little.
DECOUPLE_NS = 500


def single_qubit_error(calibration: Calibration, q: int) -> float:
    """The qubit's SX (and X) error from the calibration, or SINGLE_QUBIT_ERROR if it has none."""
    error = (calibration.single_qubit or {}).get(q)
    return SINGLE_QUBIT_ERROR if error is None else error


def idle_error(t: float, t1: float, t2: float) -> tuple[float, float]:
    """(px, pz) of an idle of t ns, Pauli-twirled from T1 and T2 decay; py = px."""
    px = (1 - exp(-t / t1)) / 4
    return px, max((1 - exp(-t / t2)) / 2 - px, 0)


def decoupled_t2(t1: float, t2: float, decoupled_dephasing: float) -> float:
    """T2 under decoupling, keeping that share of the pure dephasing."""
    return 1 / (1 / (2 * t1) + decoupled_dephasing * (1 / t2 - 1 / (2 * t1)))


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
    decoupled_dephasing: float = 0.0,
) -> stim.Circuit:
    """The flagged memory circuit on its Fez qubits, with noise, detectors and the logical.

    scale multiplies every error probability (for idling, the idle time).
    decoupling puts two X pulses, each with a single-qubit gate error, in every
    idle of at least DECOUPLE_NS. decoupled_dephasing is the share of the qubit's
    pure dephasing they leave: 0 leaves only T1 decay, 1 dephases at IBM's T2.
    IBM measures T2 with a Hahn echo, so idles without pulses flatter a bare qubit.
    """
    import stim

    circuit, schedule = memory_circuit_flagged(
        build_operators(distance), rounds=rounds, basis=basis
    )
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".* uses weak parts")
        chip = fez_qubits(distance, calibration)

    ns = calibration.durations_ns or FEZ_NS
    out = stim.Circuit()
    free: dict[int, float] = {}  # when each qubit is next free, in ns
    just_reset: set[int] = set()  # not in a CX since its reset; on the chip the reset waits instead
    records: dict[tuple[str, int], int] = {}  # (register, bit) -> measurement index

    def scaled(p: float) -> float:
        return min(scale * p, 0.5)

    def wait(q: int, until: float) -> None:
        if q in free and q not in just_reset and until > free[q]:
            idle = until - free[q]
            t1 = calibration.t1_us[chip[q]] * 1e3
            t2 = min(calibration.t2_us[chip[q]] * 1e3, 2 * t1)  # IBM's fit can exceed 2 T1
            pulses = 2 if decoupling and idle >= DECOUPLE_NS else 0
            if pulses:
                idle -= pulses * ns["x"]
                t2 = decoupled_t2(t1, t2, decoupled_dephasing)
            px, pz = idle_error(scale * idle, t1, t2)
            out.append("PAULI_CHANNEL_1", [q], [px, px, pz])
            # Pauli noise commutes with the pulses, so where they sit in the idle doesn't matter.
            for _ in range(pulses):
                out.append("X", [q])
                out.append(
                    "DEPOLARIZE1", [q], scaled(1.5 * single_qubit_error(calibration, chip[q]))
                )
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
            for q in qubits:
                out.append(
                    "DEPOLARIZE1", [q], scaled(1.5 * single_qubit_error(calibration, chip[q]))
                )
            duration = ns["x"]
        elif name == "cx":
            control, target = qubits
            error = calibration.cz[tuple(sorted((chip[control], chip[target])))]
            out.append("CX", qubits)
            # IBM quotes average gate infidelity; a depolarizing channel needs 5/4 of it.
            out.append("DEPOLARIZE2", qubits, scaled(1.25 * error))
            out.append(
                "DEPOLARIZE1", [target], scaled(3 * single_qubit_error(calibration, chip[target]))
            )
            just_reset.difference_update(qubits)
            duration = ns["cz"] + 2 * ns["x"]
        elif name == "reset":
            (q,) = qubits
            out.append("R", [q])
            out.append("X_ERROR", [q], scaled(calibration.readout[chip[q]]))
            just_reset.add(q)
            duration = ns["reset"]
        elif name == "measure":
            (q,) = qubits
            out.append("M", [q], scaled(calibration.readout[chip[q]]))
            bit = circuit.find_bit(item.clbits[0])
            register, index = bit.registers[0]
            records[register.name, index] = len(records)
            duration = ns["measure"]
        else:
            raise ValueError(f"unexpected instruction: {name}")
        for q in qubits:
            free[q] = start + duration

    add_detectors(out, schedule, records)
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
    uncertainty: float  # one standard error of per_round, widened by a poor fit
    chi2_red: float = float("nan")  # reduced chi-squared of the fit


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
    decoupled_dephasing: float = 0.0,
) -> Decay:
    """Simulate each number of rounds and fit the error per round."""
    errors = []
    for n in rounds:
        circuit = noisy_circuit(
            distance,
            basis,
            n,
            calibration,
            scale=scale,
            decoupling=decoupling,
            decoupled_dephasing=decoupled_dephasing,
        )
        errors.append(logical_errors(circuit, shots, None if seed is None else seed + n) / shots)
    return fit_per_round(rounds, errors, shots)


def fit_per_round(rounds: tuple[int, ...], logical_error: list[float], shots: int) -> Decay:
    """Fit log(1 - 2 p_L) = a + b n, each point weighted by its shot noise.

    The error per round is (1 - e^b) / 2; a takes up prep and readout error.
    Points with p_L = 0 or p_L >= 0.5 say nothing about b and are left out.
    """
    import numpy as np

    p, n = np.array(logical_error), np.array(rounds, dtype=float)
    keep = (p > 0) & (p < 0.5)
    if keep.sum() < 2:
        return Decay(tuple(rounds), tuple(logical_error), 0.5, float("nan"))
    p, n = p[keep], n[keep]
    sigma = 2 * np.sqrt(p * (1 - p) / shots) / (1 - 2 * p)  # shot noise on log(1 - 2p)
    y = np.log(1 - 2 * p)
    (b, a), cov = np.polyfit(n, y, 1, w=1 / sigma, cov="unscaled")
    # Shot noise alone, unless the points scatter more than it allows.
    chi2_red = float(np.sum(((y - a - b * n) / sigma) ** 2) / (len(n) - 2)) if len(n) > 2 else 1.0
    spread = sqrt(max(chi2_red, 1.0))
    return Decay(
        tuple(rounds),
        tuple(logical_error),
        (1 - exp(b)) / 2,
        exp(b) / 2 * sqrt(cov[0, 0]) * spread,
        chi2_red,
    )


def add_detectors(
    out: stim.Circuit,
    schedule: FlaggedSchedule,
    records: dict[tuple[str, int], int],
    observable: int = 0,
    labels: list | None = None,
) -> None:
    """Detectors on the memory-basis stabilizers, and the logical as that observable.

    records maps (register "m" or "d", bit) to its measurement in out. A detector
    is a stabilizer's value XOR its value the round before. In Z memory every
    flag and relay is a detector too: each should read 0. labels, if given, gets
    (round, stabilizer or "kind gauge") for each detector.
    """
    import stim

    patch, basis, rounds = schedule.patch, schedule.basis, schedule.rounds
    total = out.num_measurements

    def rec(register: str, bit: int) -> stim.GateTarget:
        return stim.target_rec(records[register, bit] - total)

    bits = {(m.round, m.half, m.kind, m.gauge): m.bit for m in schedule.measurements}
    stabilizers = patch.x_stabilizers if basis == "X" else patch.z_stabilizers
    kind = "x_gauge" if basis == "X" else "z_syn"

    def outcomes(r: int, name: str) -> list[stim.GateTarget]:
        return [rec("m", bits[r, basis, kind, g]) for g in patch.stabilizer_gauge_factors[name]]

    found = []
    for r in range(rounds):
        for name in stabilizers:
            previous = outcomes(r - 1, name) if r else []
            out.append("DETECTOR", outcomes(r, name) + previous)
            found.append((r, name))
    for name, support in stabilizers.items():
        data = [rec("d", q - 1) for q in sorted(support)]
        out.append("DETECTOR", data + outcomes(rounds - 1, name))
        found.append((rounds, name))
    if basis == "Z":
        for m in schedule.measurements:
            if m.kind in ("z_flag", "relay"):
                out.append("DETECTOR", [rec("m", m.bit)])
                found.append((m.round, f"{m.kind} {m.gauge}"))
    if labels is not None:
        labels += found
    logical = patch.code.logical_x if basis == "X" else patch.code.logical_z
    out.append(
        "OBSERVABLE_INCLUDE", [rec("d", q) for q in sorted(logical.x or logical.z)], observable
    )
