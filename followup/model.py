"""The calibration model of the follow-up's circuits, on the schedule they are sent with.

Stim, with Pauli channels from the day's calibration, instruction by instruction through the
scheduled native circuit (submit.py's, as sent or with a fix undone), so that every pulse, CZ,
readout and delay sits where the device runs it:

    CZ        twirled, so a Pauli channel: depolarizing, with 5/4 of IBM's error
    sx, x     depolarizing, with 3/2 of the qubit's error: H gates, the twirl's layers, XY4
    readout   the record is wrong with the qubit's readout error; the qubit is not flipped
    delay     T1 decay and dephasing, Pauli-twirled: px = py = (1 - e^(-t/T1))/4 and
              pz = (1 - e^(-t/T))/2 - px, T a data qubit's own decay time T_q, else IBM's T2

T_q is IBM's T2 before the run and the idle test's after it, so 1/T_phi = 1/T_q - 1/(2 T1).
A qubit has no noise until its first gate.

Without resets a readout leaves the qubit in |m>, and in |1> it decays at T1 until its next
gadget. The twirled channel flips it with probability (1 - e^(-t/T1))/2, that decay averaged
over m: right when m is random, as almost every readout is, and in the long idles whatever m
is, since XY4 keeps the qubit in |1> for half of each. A misread leaves the qubit as it was,
so the carry it gives the qubit's next value is wrong too: when that is a value of the same
kind, the misread fires D_t and D_(t+2). The qubit's own decay during its readout, after the
outcome is set, is not modelled: adding T1 over the whole readout raises P_L by at most 1-5%.

Run 1's design (heavyhex.experiment.circuits.for_fez) runs through the same model: a reset is
R and then a flip with the readout error, and the qubit has no noise until its next gate.
"""

from __future__ import annotations

from math import isclose, pi, sqrt
from typing import TYPE_CHECKING

from heavyhex.circuits.flagged import FlaggedSchedule
from heavyhex.simulation import noisy as run1
from heavyhex.simulation.noisy import fit_per_round, idle_error, single_qubit_error
from heavyhex.simulation.scheduled import RZ

from . import circuits, decoder
from .circuits import ROUNDS, Schedule

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    import numpy as np
    import stim
    from qiskit import QuantumCircuit

    from heavyhex.patches.placement import Calibration

SHOTS = 200_000  # simulated shots per circuit; Λ's own noise is then about 0.01
CHUNK = 25_000  # shots sampled and decoded at once
RUN_SHOTS = 8192  # per memory circuit on the chip


def noisy(
    circuit: QuantumCircuit,
    calibration: Calibration,
    dt: float,
    *,
    decay_us: Mapping[int, float] | None = None,
) -> tuple[stim.Circuit, dict[tuple[str, int], int]]:
    """The scheduled circuit with noise, and the measurement each (register, bit) lands in.

    dt is the device's time step in seconds. decay_us gives qubits their own decay time T
    (the data qubits' T_q); the others dephase at IBM's T2. Either is capped at 2 T1.
    """
    import stim

    out = stim.Circuit()
    records: dict[tuple[str, int], int] = {}
    quiet = set(range(circuit.num_qubits))  # in |0>: not yet used, or just reset
    decay_us = decay_us or {}
    for item in circuit.data:
        name, op = item.operation.name, item.operation
        qubits = [circuit.find_bit(q).index for q in item.qubits]
        if name == "barrier":
            out.append("TICK", [])
        elif name == "delay":
            (q,) = qubits
            if q in quiet:
                continue
            t1 = calibration.t1_us[q] * 1e3
            t2 = min(decay_us.get(q, calibration.t2_us[q]) * 1e3, 2 * t1)
            px, pz = idle_error(_ns(op, dt), t1, t2)
            out.append("PAULI_CHANNEL_1", [q], [px, px, pz])
        elif name == "rz":
            quarters = float(op.params[0]) / (pi / 2)
            if not isclose(quarters, round(quarters), abs_tol=1e-6):  # samplomatic's are float32
                raise ValueError(f"rz({op.params[0]}) is not a Clifford gate")
            gate = RZ[round(quarters) % 4]
            if gate:
                out.append(gate, qubits)
        elif name in ("sx", "x"):
            out.append("SQRT_X" if name == "sx" else "X", qubits)
            # IBM quotes average gate infidelity; a depolarizing channel needs 3/2 of it.
            out.append(
                "DEPOLARIZE1", qubits, min(1.5 * single_qubit_error(calibration, *qubits), 0.75)
            )
            quiet.difference_update(qubits)
        elif name == "cz":
            error = calibration.cz.get((min(qubits), max(qubits)))
            if error is None:
                raise ValueError(f"no CZ error for {qubits}: IBM marks the coupler broken")
            out.append("CZ", qubits)
            out.append("DEPOLARIZE2", qubits, min(1.25 * error, 15 / 16))  # 5/4 likewise
            quiet.difference_update(qubits)
        elif name == "measure":
            (q,) = qubits
            out.append("M", [q], calibration.readout[q])
            register, index = circuit.find_bit(item.clbits[0]).registers[0]
            records[register.name, index] = out.num_measurements - 1
        elif name == "reset":
            out.append("R", qubits)
            out.append("X_ERROR", qubits, calibration.readout[qubits[0]])
            quiet.update(qubits)
        else:
            raise ValueError(f"unexpected instruction: {name}")
    return out, records


def memory(
    circuit: QuantumCircuit,
    schedule: Schedule | FlaggedSchedule,
    calibration: Calibration,
    dt: float,
    *,
    decay_us: Mapping[int, float] | None = None,
) -> tuple[stim.Circuit, list[tuple[int, str]]]:
    """A memory circuit's model with its detectors and logical, and the detectors' labels.

    A follow-up Schedule gets the detectors without resets (circuits.add_detectors), run 1's
    FlaggedSchedule run 1's own.
    """
    model, records = noisy(circuit, calibration, dt, decay_us=decay_us)
    own = {(name[0], i): k for (name, i), k in records.items()}  # "m5" -> "m", as in run 1
    labels: list[tuple[int, str]] = []
    if isinstance(schedule, Schedule):
        circuits.add_detectors(model, schedule, own, labels=labels)
    else:
        run1.add_detectors(model, schedule, own, labels=labels)
    return model, labels


def decay_times(
    calibration: Calibration, data: Sequence[int], measured: Mapping[int, float] | None = None
) -> dict[int, float]:
    """Each data qubit's T_q in us: the idle test's where measured, else IBM's T2."""
    return {q: (measured or {}).get(q, calibration.t2_us[q]) for q in data}


def sample(
    model: stim.Circuit,
    decoders: Mapping[str, Callable[[np.ndarray], np.ndarray]],
    shots: int = SHOTS,
    seed: int = 0,
) -> tuple[dict[str, float], np.ndarray]:
    """The share of shots each decoder gets wrong, and each detector's rate."""
    import numpy as np

    sampler = model.compile_detector_sampler(seed=seed)
    failed, fired = dict.fromkeys(decoders, 0), np.zeros(model.num_detectors)
    for start in range(0, shots, CHUNK):
        detectors, flips = sampler.sample(min(CHUNK, shots - start), separate_observables=True)
        for name, decode in decoders.items():
            failed[name] += int(np.count_nonzero(decode(detectors)[:, 0] != flips[:, 0]))
        fired += detectors.sum(axis=0)
    return {name: n / shots for name, n in failed.items()}, fired / shots


def detection_rates(labels: Sequence[tuple[int, str]], rates: np.ndarray, rounds: int) -> dict:
    """The mean rate of each kind of detector: stabilizers in the rounds, the final
    comparison with the data, flags and relays."""
    import numpy as np

    kinds: dict[str, list[float]] = {}
    for (r, name), rate in zip(labels, rates, strict=True):
        kind = name.split()[0] if " " in name else "final" if r == rounds else "stabilizer"
        kinds.setdefault(kind, []).append(float(rate))
    return {kind: float(np.mean(found)) for kind, found in kinds.items()}


def fits(logical_error: Mapping[tuple[str, int, int], float], shots: int) -> dict:
    """ε per round for each basis and distance (run 1's weighted fit of ln(1 - 2P) on the
    rounds), and Λ = ε₃/ε₅. logical_error is keyed by (basis, distance, rounds)."""
    found: dict = {}
    for basis in "XZ":
        per_round = {}
        for d in circuits.DISTANCES:
            fit = fit_per_round(ROUNDS, [logical_error[basis, d, n] for n in ROUNDS], shots)
            per_round[d] = (fit.per_round, fit.uncertainty)
        (e3, u3), (e5, u5) = per_round[3], per_round[5]
        lam = e3 / e5
        found[basis] = {
            "eps": per_round,
            "lambda": (lam, lam * sqrt((u3 / e3) ** 2 + (u5 / e5) ** 2)),
        }
    return found


def predict(
    circuits: Mapping[tuple[str, int, int], tuple[QuantumCircuit, Schedule | FlaggedSchedule]],
    calibration: Calibration,
    dt: float,
    *,
    decay_us: Mapping[int, float] | None = None,
    shots: int = SHOTS,
    seed: int = 0,
    decoders: Sequence[float | str | None] = (*decoder.CAPS, decoder.CORRELATED),
) -> dict:
    """What the model expects of the memory circuits, keyed by (basis, distance, rounds).

    Returns the rule's cap (the first of decoder.CAPS that corrects every single fault of
    every circuit) and the single faults each cap miscorrects; each circuit's mean detection
    rates; and, for each of `decoders` by str(cap), each circuit's logical error, the fits at
    the simulated shots and their standard errors at RUN_SHOTS. The rule's decoder's are also
    at the top, when it is among them.
    """
    keys = sorted(circuits)
    models = {key: memory(*circuits[key], calibration, dt, decay_us=decay_us) for key in keys}
    dems = {key: models[key][0].detector_error_model(decompose_errors=True) for key in keys}
    cap, counts = decoder.capped(dems[key] for key in keys)
    logical_error: dict[str, dict] = {str(d): {} for d in decoders}
    detection = {}
    for i, key in enumerate(keys):
        model, labels = models[key]
        p, rates = sample(model, decoder.decoders(dems[key], decoders), shots, seed + i)
        for name, value in p.items():
            logical_error[name][key] = value
        detection[key] = detection_rates(labels, rates, key[2])
    by_decoder = {
        name: {"logical_error": p, "fits": fits(p, shots), "precision": fits(p, RUN_SHOTS)}
        for name, p in logical_error.items()
    }
    return {
        "cap": cap,
        "miscorrected": {
            str(c): dict(zip(map(str, keys), n, strict=True)) for c, n in counts.items()
        },
        "detection": detection,
        "decoders": by_decoder,
        **by_decoder.get(str(cap), {}),
    }


def _ns(op, dt: float) -> float:
    """A delay's length in ns."""
    if op.unit == "dt":
        return op.duration * dt * 1e9
    scale = {"s": 1e9, "ms": 1e6, "us": 1e3, "ns": 1.0}
    if op.unit not in scale:
        raise ValueError(f"unexpected delay unit: {op.unit}")
    return op.duration * scale[op.unit]
