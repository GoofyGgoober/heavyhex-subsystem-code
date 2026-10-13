"""The follow-up's analysis, from the jobs' records in runs/.

Exp 1, both jobs together:

    decode   each memory circuit's detectors from its raw readouts (a value is its readout
             XOR its carry), matched on the model of the circuit as sent (program.json.gz);
             weights capped at 0.1 lead (decoder.PRIMARY), uniform weights beside them
    fit      ε per round from P(n) = [1 - A (1 - 2ε)^n] / 2: run 1's weighted fit of
             ln(1 - 2P), and the binomial likelihood with A profiled out, its interval widened
             by the Pearson dispersion; Λ = ε₃/ε₅, the errors propagated
    T_q      per data qubit, r = √(⟨X⟩² + ⟨Y⟩²) over the idle lengths, fitted to
             ln r = c + ln G(n) - t(n)/T_q, weighted by
             Var(ln r) = [x²(1 - x²)/N_X + y²(1 - y²)/N_Y] / r⁴;
             t is the qubit's delay time on the equator (T1 alone decays it along Z, between
             a wait's H gates) and G what the model's pulses, readout and T1 leave of r, so
             T_q is on the model's clock and goes straight into it
    model    the model again at the measured T_q: O/M per basis and distance, the detection
             rate of each kind of detector, and each flag's and relay's record
    run 1    its fifteen jobs decoded alike and fitted on rounds 1-4, each modelled as run 1's
             analysis modelled it; per placement, ε the mean over the days and Λ the
             inverse-variance mean of the days', widened by √(χ²/dof) above 1

life-1:

    T1, T2   every qubit of the footprint, P(1) = A e^(-t/T) + B (lifetimes.decay), compared
             in the rate 1/T; T1's B gives the excited share at rest, π
    Z idle   each data qubit's flips per round, p↓ + p↑ (lifetimes.flips), split as π says,
             and the contrast c = P(1 | 1) - P(1 | 0) over the idle lengths, weighted by
             Var(ln c) = [p1(1 - p1)/N1 + p0(1 - p0)/N0] / c²
    in situ  each data qubit's T1 and T_q from both idle tests at once, on the model's clock:
             ln r = k_X + ln G(n) - t_eq(n)/T_q - t_Z(n)/T1 read in X and Y (Exp 1), and
             ln c = k_Z + ln G(n) - t_Z(n)/T1 - t_eq(n)/T_q read in Z, t_eq and t_Z the
             delay time on the equator and along Z, G what the model's pulses and readout
             leave. A data qubit spends about 2.9 us of a round on one and 2.3 us on the
             other, swapped between the tests, so neither rate comes from one test alone;
             what they pin down is the decay a round each memory's data see,
             t_Z/T1 + t_eq/T_q with each memory's t_Z and t_eq
    compare  each qubit's T1 and T2: IBM's in force, ours, and in situ (lifetimes.table)
    model    Exp 1's memory again at the data's in-situ T1 and T_q, and O/M against it

Exp 3: each job's readouts and the cycle period, for leakage_control.analyze.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from math import exp, pi, sqrt
from statistics import median
from typing import TYPE_CHECKING

import numpy as np

from heavyhex.patches.placement import Calibration
from heavyhex.simulation.noisy import fit_per_round

from . import circuits, decoder, lifetimes, model, submit
from . import leakage_control as exp3
from .circuits import DISTANCES, ROUNDS, Setting, read_into
from .placement import place, role_of

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import stim
    from qiskit import QuantumCircuit

MIN_R = 0.05  # an idle length whose r (or contrast) is below this says nothing about T_q (T1)
FITS = ("log", "binomial")


# The records


def sent(folder: Path) -> dict[str, tuple[Setting | None, QuantumCircuit]]:
    """Each circuit of a job as it was sent, by label: its setting, and the circuit with one
    randomization of its twirl drawn (they share one model)."""
    from qiskit_ibm_runtime.quantum_program.params_converters import (
        QUANTUM_PROGRAM_PARAMS_CONVERTERS,
    )

    with gzip.open(folder / "program.json.gz", "rt") as f:
        payload = json.load(f)
    converter = QUANTUM_PROGRAM_PARAMS_CONVERTERS[payload["schema_version"]]
    program, _ = converter.decoder(converter.model.model_validate(payload))
    rows = json.loads((folder / "run.json").read_text())["items"]
    found = {}
    for row, item in zip(rows, program.items, strict=True):
        if row["label"] in found:
            continue
        samplex = getattr(item, "samplex", None)
        if samplex is not None:
            samplex.finalize()
        one = submit.Item(row["label"], item.circuit, row["seconds"], row["shots"], samplex, 1)
        setting = Setting(**row["setting"]) if row["setting"] else None
        found[row["label"]] = (setting, submit.instances(one, 1)[0])
    return found


def readouts(folders: Sequence[Path]) -> dict[str, dict[str, np.ndarray]]:
    """Each circuit's readouts by register, shots x bits, the jobs' shots in a row."""
    found: dict[str, dict[str, list[np.ndarray]]] = defaultdict(lambda: defaultdict(list))
    for folder in folders:
        for label, registers in submit.load(folder).items():
            for name, values in registers.items():
                found[label][name].append(values)
    return {
        label: {name: np.concatenate(parts) for name, parts in registers.items()}
        for label, registers in found.items()
    }


# Decoding


def memory_model(
    circuit: QuantumCircuit,
    schedule: circuits.Schedule,
    calibration: Calibration,
    dt: float,
    *,
    decay_us: Mapping[int, float] | None = None,
    flags: bool | None = None,
) -> tuple[stim.Circuit, dict[tuple[str, int], int], list[tuple[int, str]]]:
    """The circuit's model with its detectors and logical, the measurement each (register,
    bit) lands in, and the detectors' labels."""
    noisy, records = model.noisy(circuit, calibration, dt, decay_us=decay_us)
    own = {(name[0], i): k for (name, i), k in records.items()}  # "m5" -> "m"
    labels: list[tuple[int, str]] = []
    circuits.add_detectors(noisy, schedule, own, labels=labels, flags=flags)
    return noisy, records, labels


def detectors(
    noisy: stim.Circuit, records: Mapping[tuple[str, int], int], bits: Mapping[str, np.ndarray]
) -> tuple[np.ndarray, np.ndarray]:
    """The detectors and logical of each shot, from its readouts by register."""
    shots = len(next(iter(bits.values())))
    measured = np.zeros((shots, len(records)), dtype=bool)
    for (name, i), k in records.items():
        measured[:, k] = bits[name][:, i]
    converter = noisy.compile_m2d_converter(skip_reference_sample=True)
    return converter.convert(measurements=measured, separate_observables=True)


def memory(
    setting: Setting,
    circuit: QuantumCircuit,
    bits: Mapping[str, np.ndarray],
    calibration: Calibration,
    dt: float,
    *,
    decay_us: Mapping[int, float] | None,
    caps: Sequence[float | str | None],
) -> dict:
    """One memory circuit: its shots, the shots each decoder gets wrong (by str(cap)), and the
    mean detection rate of each kind of detector."""
    _, schedule = circuits.logical_circuit(setting)
    noisy, records, labels = memory_model(circuit, schedule, calibration, dt, decay_us=decay_us)
    fired, flips = detectors(noisy, records, bits)
    dem = noisy.detector_error_model(decompose_errors=True)
    wrong = {
        name: int(np.count_nonzero(decode(fired)[:, 0] != flips[:, 0]))
        for name, decode in decoder.decoders(dem, caps).items()
    }
    return {
        "shots": len(fired),
        "failures": wrong,
        "detection": model.detection_rates(labels, fired.mean(axis=0), setting.rounds),
    }


# Fits


@dataclass(frozen=True)
class Binomial:
    """ε by maximum likelihood, k_n ~ Binomial(N, P(n)), and its interval: Δ(-2 ln L) of the
    Pearson dispersion φ = X²/(m - 2) (1 if below), the profile over A."""

    eps: float
    amplitude: float
    low: float
    high: float
    dispersion: float

    @property
    def sigma(self) -> float:
        return (self.high - self.low) / 2


def binomial(rounds: Sequence[int], failures: Sequence[float] | np.ndarray, shots: int) -> Binomial:
    """The binomial fit of P(n) = [1 - A (1 - 2ε)^n] / 2, A profiled out (concave in A)."""
    from scipy.optimize import minimize_scalar, root_scalar

    n, k = np.asarray(rounds, float), np.asarray(failures, float)
    top_eps = 0.5 - 1e-9

    def loglik(eps: float, a: float) -> float:
        p = np.clip((1 - a * (1 - 2 * eps) ** n) / 2, 1e-15, 1 - 1e-15)
        return float(k @ np.log(p) + (shots - k) @ np.log1p(-p))

    def best(eps: float) -> tuple[float, float]:
        q = 1 - 2 * eps
        found = minimize_scalar(
            lambda a: -loglik(eps, a),
            bounds=(0, (1 - 1e-12) / q),
            method="bounded",
            options={"xatol": 1e-12},
        )
        return float(found.x), -float(found.fun)

    lo, hi = 1e-5, top_eps
    for _ in range(6):  # a grid, then finer grids around its best point
        grid = np.linspace(lo, hi, 41)
        i = int(np.argmax([best(e)[1] for e in grid]))
        lo, hi = float(grid[max(i - 1, 0)]), float(grid[min(i + 1, 40)])
    eps = (lo + hi) / 2
    a, top = best(eps)
    p = (1 - a * (1 - 2 * eps) ** n) / 2
    dispersion = max(
        float(((k - shots * p) ** 2 / (shots * p * (1 - p))).sum()) / (len(n) - 2), 1.0
    )

    def edge(lo: float, hi: float, default: float) -> float:
        def gap(e: float) -> float:
            return 2 * (top - best(e)[1]) - dispersion

        if gap(lo) * gap(hi) >= 0:
            return default
        return float(root_scalar(gap, bracket=(lo, hi), xtol=1e-12).root)

    return Binomial(eps, a, edge(1e-9, eps, 0.0), edge(eps, top_eps, 0.5), dispersion)


def curve(failures: Sequence[int], shots: int, rounds: Sequence[int] = ROUNDS) -> dict:
    """Both fits of one curve: [ε, σ] each, the binomial interval and dispersion."""
    log = fit_per_round(tuple(rounds), [k / shots for k in failures], shots)
    b = binomial(rounds, failures, shots)
    return {
        "failures": list(failures),
        "log": [log.per_round, log.uncertainty],
        "binomial": [b.eps, b.sigma],
        "binomial_interval": [b.low, b.high],
        "dispersion": b.dispersion,
    }


def ratio(a: Sequence[float], b: Sequence[float]) -> list[float]:
    """a/b and its standard error, for independent [value, σ]."""
    (x, u), (y, v) = a, b
    r = x / y
    return [r, r * sqrt((u / x) ** 2 + (v / y) ** 2)]


def fits(failures: Mapping[tuple[str, int, int], int], shots: int) -> dict:
    """ε per round for each basis and distance there is, by both fits, and Λ by each.
    failures is keyed by (basis, distance, rounds)."""
    found: dict = {}
    for basis in "XZ":
        eps = {
            d: curve([failures[basis, d, n] for n in ROUNDS], shots)
            for d in DISTANCES
            if (basis, d, ROUNDS[0]) in failures
        }
        found[basis] = {"eps": eps}
        if len(eps) == 2:
            found[basis]["lambda"] = {fit: ratio(eps[3][fit], eps[5][fit]) for fit in FITS}
    return found


# The idle test


def axis_times(
    circuit: QuantumCircuit, qubits: Sequence[int], dt: float
) -> dict[str, dict[int, float]]:
    """Each qubit's delay time in us up to its readout, with its state on the equator and
    along Z, from its first pulse on (before it the model has the qubit quiet in |0>).

    In the idle test a data qubit gets one-qubit gates only, so its state stays along X, Y or
    Z. Each delay takes e^(-t/T) of its ⟨X⟩ on the equator, and e^(-t/T1) of its ⟨Z⟩ along Z:
    read in X or Y only the equator's time measures T_q, read in Z only Z's measures T1.
    """
    swap = {"rz": {"X": "Y", "Y": "X"}, "sx": {"Y": "Z", "Z": "Y"}}
    axis = dict.fromkeys(qubits, "Z")
    started: set[int] = set()
    total = {"equator": dict.fromkeys(qubits, 0.0), "Z": dict.fromkeys(qubits, 0.0)}
    for item in circuit.data:
        op = item.operation
        for q in (circuit.find_bit(bit).index for bit in item.qubits):
            if axis.get(q, "read") == "read":
                continue
            if op.name == "measure":
                axis[q] = "read"
            elif op.name == "delay":
                if q in started:
                    total["Z" if axis[q] == "Z" else "equator"][q] += model._ns(op, dt) / 1e3
            elif op.name in ("sx", "x", "rz"):
                if op.name != "rz":
                    started.add(q)
                if op.name == "sx" or (
                    op.name == "rz" and round(float(op.params[0]) / (pi / 2)) % 2
                ):
                    axis[q] = swap[op.name].get(axis[q], axis[q])
            elif op.name != "barrier":
                raise ValueError(f"{op.name} on qubit {q}: the idle test has one-qubit gates only")
    return total


def pulses(circuit: QuantumCircuit, calibration: Calibration, dt: float) -> np.ndarray:
    """G: what the model's pulses and readout leave of each data qubit's contrast, by data bit.
    The product of 1 - 2p over the errors that flip its readout, times e^(t/T) e^(t'/T1) for
    the decay its delays take: e^(-t/T) of its time t on the equator (T = IBM's T2, at most
    2 T1, as the model has it) and e^(-t'/T1) of its time t' along Z, exactly."""
    import stim

    noisy, records = model.noisy(circuit, calibration, dt)
    read = read_into(circuit, "d")
    register = next(name for name, _ in records if name.startswith("d"))
    for k, i in enumerate(read):
        noisy.append(
            "OBSERVABLE_INCLUDE",
            [stim.target_rec(records[register, i] - noisy.num_measurements)],
            k,
        )
    x = np.ones(len(read))
    for line in decoder._lines(noisy.detector_error_model()):
        if line.type == "error":
            for target in line.targets_copy():
                if isinstance(target, stim.DemTarget) and target.is_logical_observable_id():
                    x[target.val] *= 1 - 2 * line.args_copy()[0]
    times = axis_times(circuit, list(read.values()), dt)
    for k, q in enumerate(read.values()):
        t1 = calibration.t1_us[q]
        t2 = min(calibration.t2_us[q], 2 * t1)
        x[k] *= exp(times["equator"][q] / t2 + times["Z"][q] / t1)
    return x


def r_of(
    x: np.ndarray, y: np.ndarray, nx: np.ndarray, ny: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """r = √(⟨X⟩² + ⟨Y⟩²), r² less the shot noise in x² and y², and
    Var(ln r) = [x²(1 - x²)/N_X + y²(1 - y²)/N_Y] / r⁴."""
    r = np.sqrt(np.maximum(x**2 + y**2 - (1 - x**2) / nx - (1 - y**2) / ny, 0))
    return r, (x**2 * (1 - x**2) / nx + y**2 * (1 - y**2) / ny) / np.maximum(r, MIN_R) ** 4


def c_of(
    p0: np.ndarray, p1: np.ndarray, n0: np.ndarray, n1: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The contrast c = P(1 | 1) - P(1 | 0), and Var(ln c) = [p1(1 - p1)/N1 + p0(1 - p0)/N0] / c²."""
    c = p1 - p0
    return c, (p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0) / np.maximum(c, MIN_R) ** 2


def log_decay(t: np.ndarray, value: np.ndarray, var: np.ndarray, g: np.ndarray) -> dict:
    """T and its error from ln(value/G) = c - t/T, weighted by 1/Var(ln value), over the
    points whose value is above MIN_R, and the rate 1/T in 1/us with its error; NaN with
    fewer than two, inf if nothing decays."""
    keep = value > MIN_R
    nan = float("nan")
    if keep.sum() < 2:
        return {
            "t_us": nan,
            "sigma_us": nan,
            "rate": [nan, nan],
            "offset": nan,
            "chi2_red": nan,
            "used": int(keep.sum()),
        }
    t, ln, var = t[keep], np.log(value[keep] / g[keep]), var[keep]
    (b, c), cov = np.polyfit(t, ln, 1, w=1 / np.sqrt(var), cov="unscaled")
    chi2 = float((((ln - c - b * t) ** 2) / var).sum())
    return {
        "t_us": float(-1 / b) if b < 0 else float("inf"),
        "sigma_us": float(sqrt(cov[0, 0]) / b**2),
        "rate": [float(-b), float(sqrt(cov[0, 0]))],
        "offset": float(exp(c)),
        "chi2_red": chi2 / (keep.sum() - 2) if keep.sum() > 2 else nan,
        "used": int(keep.sum()),
    }


def decay_time(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, nx: np.ndarray, ny: np.ndarray, g: np.ndarray
) -> dict:
    """T_q and its error from one qubit's ⟨X⟩ and ⟨Y⟩ over the idle lengths: ln(r/G) = c - t/T_q,
    weighted. NaN with fewer than two r above MIN_R."""
    fit = log_decay(t, *r_of(x, y, nx, ny), g)
    return {"t_q_us": fit.pop("t_us"), **fit}


def contrast_time(
    t: np.ndarray, p0: np.ndarray, p1: np.ndarray, n0: np.ndarray, n1: np.ndarray, g: np.ndarray
) -> dict:
    """T1 and its error from one data qubit's idle test read in Z: ln(c/G) = k - t/T1 over the
    idle lengths, weighted. NaN with fewer than two c above MIN_R."""
    fit = log_decay(t, *c_of(p0, p1, n0, n1), g)
    return {"t1_us": fit.pop("t_us"), **fit}


def idle(
    found: Mapping[str, tuple[Setting | None, QuantumCircuit]],
    shots: Mapping[str, Mapping[str, np.ndarray]],
    calibration: Calibration,
    dt: float,
) -> dict:
    """Each data qubit's T_q from the idle test read in X and Y, its G holding the decay along
    Z at IBM's T1, and the decay times the model takes: the measured T_q where finite, else
    IBM's T2."""
    runs: dict[str, dict[int, str]] = {"X": {}, "Y": {}}
    for label, (setting, _) in found.items():
        if setting is not None and setting.kind == "idle" and setting.readout in runs:
            runs[setting.readout][setting.rounds] = label
    rounds = sorted(runs["X"])
    if sorted(runs["Y"]) != rounds:
        raise ValueError("the idle test needs the same lengths read in X and in Y")
    times, g, read = _waits([found[runs["X"][n]][1] for n in rounds], calibration, dt)
    register = next(name for name in shots[runs["X"][rounds[0]]] if name.startswith("d"))
    x, y, nx, ny = ([] for _ in range(4))
    for n in rounds:
        bx, by = shots[runs["X"][n]][register], shots[runs["Y"][n]][register]
        x.append(1 - 2 * bx.mean(axis=0))
        y.append(1 - 2 * by.mean(axis=0))
        nx.append(len(bx))
        ny.append(len(by))
    x_all, y_all = np.array(x), np.array(y)
    n_x, n_y = np.array(nx, float), np.array(ny, float)
    rows = []
    for k, (i, q) in enumerate(read.items()):
        equator, along_z = times["equator"][:, k], times["Z"][:, k]
        held = g[:, k] * np.exp(-along_z / calibration.t1_us[q])
        rows.append(
            {
                "data": i,
                "qubit": q,
                "t1_us": calibration.t1_us[q],
                "t2_us": calibration.t2_us[q],
                **decay_time(equator, x_all[:, k], y_all[:, k], n_x, n_y, held),
                "equator_us": equator.tolist(),
                "along_z_us": along_z.tolist(),
                "pulses": g[:, k].tolist(),
                "x": x_all[:, k].tolist(),
                "y": y_all[:, k].tolist(),
                "shots_x": nx,
                "shots_y": ny,
            }
        )
    decay = {  # no decay seen: inf, which the model caps at 2 T1
        row["qubit"]: calibration.t2_us[row["qubit"]] if np.isnan(row["t_q_us"]) else row["t_q_us"]
        for row in rows
    }
    return {"rounds": rounds, "qubits": rows, "decay_us": decay}


def z_idle(
    found: Mapping[str, tuple[Setting | None, QuantumCircuit]],
    shots: Mapping[str, Mapping[str, np.ndarray]],
    calibration: Calibration,
    dt: float,
) -> dict:
    """Each data qubit's flips per round (lifetimes.flips) and T1 from the idle test read in
    Z, its G holding the decay on the equator at IBM's T2. The times are those of the circuit
    from |1>, whose X is the data's first pulse; from |0> the data wait as long, but quiet in
    |0> until their next pulse."""
    runs: dict[str, dict[int, str]] = {"0": {}, "1": {}}
    for label, (setting, _) in found.items():
        if setting is not None and setting.kind == "idle" and setting.readout == "Z":
            runs[setting.start][setting.rounds] = label
    rounds = sorted(runs["1"])
    if not rounds or sorted(runs["0"]) != rounds:
        raise ValueError("the idle test read in Z needs the same lengths from |0> and |1>")
    times, g, read = _waits([found[runs["1"][n]][1] for n in rounds], calibration, dt)
    register = next(name for name in shots[runs["1"][rounds[0]]] if name.startswith("d"))
    ones = {s: np.array([shots[runs[s][n]][register].mean(axis=0) for n in rounds]) for s in "01"}
    counts = {s: [len(shots[runs[s][n]][register]) for n in rounds] for s in "01"}
    n0, n1 = (np.array(counts[s], float) for s in "01")
    rows = []
    for k, (i, q) in enumerate(read.items()):
        p0, p1 = ones["0"][:, k], ones["1"][:, k]
        equator, along_z = times["equator"][:, k], times["Z"][:, k]
        held = g[:, k] * np.exp(-equator / min(calibration.t2_us[q], 2 * calibration.t1_us[q]))
        rows.append(
            {
                "data": i,
                "qubit": q,
                **contrast_time(along_z, p0, p1, n0, n1, held),
                "per_round": lifetimes.flips(rounds, p0, p1, n0, n1, calibration.readout[q]),
                "equator_us": equator.tolist(),
                "along_z_us": along_z.tolist(),
                "pulses": g[:, k].tolist(),
                "p0": p0.tolist(),
                "p1": p1.tolist(),
                "shots_0": counts["0"],
                "shots_1": counts["1"],
            }
        )
    return {"rounds": rounds, "qubits": rows}


def _waits(
    by_length: Sequence[QuantumCircuit], calibration: Calibration, dt: float
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[int, int]]:
    """An idle test's data over its lengths: their delay times on the equator and along Z
    (lengths x data bits), the model's G, and the qubit read into each data bit."""
    read = read_into(by_length[0], "d")
    times: dict[str, list] = {"equator": [], "Z": []}
    g = []
    for circuit in by_length:
        found = axis_times(circuit, list(read.values()), dt)
        for side, rows in times.items():
            rows.append([found[side][q] for q in read.values()])
        g.append(pulses(circuit, calibration, dt))
    return {side: np.array(rows) for side, rows in times.items()}, np.array(g), read


def together(xy: Mapping, z: Mapping) -> dict:
    """One data qubit's T1 and T_q from both idle tests at once: least squares weighted by
    1/Var(ln), over the points above MIN_R, of

        ln(r/G) = k_X - t_eq/T_q - t_Z/T1     read in X and Y (a row of idle(), Exp 1)
        ln(c/G) = k_Z - t_Z/T1 - t_eq/T_q     read in Z (a row of z_idle(), life-1)

    with G what the model's pulses and readout leave, in the rates the model can take:
    1/T1 >= 0 and 1/T_q = 1/(2 T1) + 1/T_phi with 1/T_phi >= 0. The data sit on the equator
    in one test about as long as along Z in the other, so the tests pin down the decay each
    memory sees, the rates' sums, much better than T1 and T_q apart: their correlation is
    near -1. The rates 1/T1 and 1/T_q in 1/us, each [value, σ], and their covariance, from
    the unbounded fit's; NaN with fewer than two points in either test."""
    from scipy.optimize import lsq_linear

    r, var_r = r_of(*(np.asarray(xy[k], float) for k in ("x", "y", "shots_x", "shots_y")))
    c, var_c = c_of(*(np.asarray(z[k], float) for k in ("p0", "p1", "shots_0", "shots_1")))
    design, ln, weight, used = [], [], [], []
    for which, (value, var, row) in enumerate(((r, var_r, xy), (c, var_c, z))):
        keep = value > MIN_R
        g, equator, along_z = (
            np.asarray(row[k], float)[keep] for k in ("pulses", "equator_us", "along_z_us")
        )
        # ln = k - (1/T1)(t_Z + t_eq/2) - (1/T_phi) t_eq
        design += [
            [1.0 - which, float(which), -tz - te / 2, -te]
            for tz, te in zip(along_z, equator, strict=True)
        ]
        ln += list(np.log(value[keep] / g))
        weight += list(1 / var[keep])
        used.append(int(keep.sum()))
    nan = float("nan")
    if min(used) < 2:
        return {"rate_t1": [nan, nan], "rate_t_q": [nan, nan], "covariance": [[nan] * 2] * 2,
                "correlation": nan, "chi2_red": nan, "used": used}  # fmt: skip
    a, y, w = np.array(design), np.array(ln), np.array(weight)
    root = np.sqrt(w)
    fit = lsq_linear(a * root[:, None], y * root, bounds=([-np.inf, -np.inf, 0, 0], np.inf))
    beta = np.where(fit.active_mask == -1, 0.0, fit.x)  # a rate on its bound is 0
    cov = np.linalg.inv((a.T * w) @ a)
    chi2 = float((w * (y - a @ beta) ** 2).sum())
    # From (1/T1, 1/T_phi) to (1/T1, 1/T_q = 1/(2 T1) + 1/T_phi).
    to_q = np.array([[1.0, 0.0], [0.5, 1.0]])
    rates, covariance = to_q @ beta[2:], to_q @ cov[2:, 2:] @ to_q.T
    return {
        "rate_t1": [float(rates[0]), float(sqrt(covariance[0, 0]))],
        "rate_t_q": [float(rates[1]), float(sqrt(covariance[1, 1]))],
        "covariance": covariance.tolist(),
        "correlation": float(covariance[0, 1] / sqrt(covariance[0, 0] * covariance[1, 1])),
        "chi2_red": chi2 / (len(y) - 4) if len(y) > 4 else nan,
        "used": used,
    }


def relaxation(
    found: Mapping[str, tuple[Setting | None, QuantumCircuit]],
    shots: Mapping[str, Mapping[str, np.ndarray]],
    calibration: Calibration,
    dt: float,
) -> dict[str, dict[int, dict]]:
    """T1 and T2 of every qubit read, by kind and qubit: lifetimes.decay's fit at the waits
    of the circuits as sent, from the calibration's value, with the waits and the share of
    readouts that read 1 at each."""
    labels: dict[str, dict[float, str]] = {kind: {} for kind in lifetimes.WAITS_US}
    for label, (_, circuit) in found.items():
        if (kind := lifetimes.kind_of(label)) is not None:
            labels[kind][lifetimes.wait_us(circuit, dt)] = label
    fitted: dict[str, dict[int, dict]] = {}
    for kind, by_wait in labels.items():
        waits = sorted(by_wait)
        read = read_into(found[by_wait[waits[0]]][1], "m")
        bits = [shots[by_wait[t]]["m"] for t in waits]
        (n,) = {len(b) for b in bits}
        ones = np.array([b.sum(axis=0) for b in bits])
        guess = calibration.t1_us if kind == "T1" else calibration.t2_us
        fitted[kind] = {
            q: {
                **lifetimes.decay(waits, ones[:, i], n, guess[q]),
                "waits_us": waits,
                "ones": (ones[:, i] / n).tolist(),
            }
            for i, q in read.items()
        }
    return fitted


@dataclass(frozen=True)
class Record:
    """A job's circuits as sent and its readouts, with its calibration and time step."""

    found: Mapping[str, tuple[Setting | None, QuantumCircuit]]
    shots: Mapping[str, Mapping[str, np.ndarray]]
    calibration: Calibration
    dt: float


def record(folders: Sequence[Path]) -> Record:
    """The jobs' record: the first one's circuits, calibration and dt, every job's shots."""
    run = json.loads((folders[0] / "run.json").read_text())
    calibration = Calibration.load(folders[0] / "calibration.json")
    return Record(sent(folders[0]), readouts(folders), calibration, run["dt"])


def in_situ(xy: Record, z: Record) -> dict:
    """The data qubits' T1 and T_q in situ, Exp 1's idle test (xy) and life-1's (z) fitted
    together, with the decay a round each memory's data see, t_Z/T1 + t_eq/T_q with the
    times a round of the test that mimics it (Z memory's read in Z), at those rates and at
    IBM's T1 and T2 as Exp 1's model took them; and the values the model takes: the fit's,
    inf (no decay) included, where there is one, else IBM's T1 and T2."""
    calibration = xy.calibration
    read_xy = idle(xy.found, xy.shots, calibration, xy.dt)
    read_z = z_idle(z.found, z.shots, z.calibration, z.dt)
    by_xy = {row["qubit"]: row for row in read_xy["qubits"]}
    by_z = {row["qubit"]: row for row in read_z["qubits"]}
    rows = []
    for q in sorted(set(by_xy) & set(by_z)):
        fit = together(by_xy[q], by_z[q])
        rates = np.array([fit["rate_t1"][0], fit["rate_t_q"][0]])
        cov = np.array(fit["covariance"])
        t1 = calibration.t1_us[q]
        ibm = np.array([1 / t1, 1 / min(calibration.t2_us[q], 2 * t1)])
        for memory, row, rounds in (
            ("z", by_z[q], read_z["rounds"]),
            ("x", by_xy[q], read_xy["rounds"]),
        ):
            times = _per_round(row, rounds)
            fit[f"decay_{memory}"] = [
                float(times @ rates),
                float(sqrt(max(times @ cov @ times, 0))),
            ]
            fit[f"decay_{memory}_ibm"] = float(times @ ibm)
        rows.append({"qubit": q, **fit})

    def taken(name: str, ibm: Mapping[int, float]) -> dict[int, float]:
        found = {}
        for row in rows:
            value = row[name][0]
            found[row["qubit"]] = (
                ibm[row["qubit"]] if np.isnan(value) else 1 / value if value > 0 else float("inf")
            )
        return found

    return {
        "qubits": rows,
        "t1_us": taken("rate_t1", calibration.t1_us),
        "decay_us": taken("rate_t_q", calibration.t2_us),
    }


def _per_round(row: Mapping, rounds: Sequence[int]) -> np.ndarray:
    """A data qubit's delay time a round along Z and on the equator, in us: the slopes over
    the idle test's lengths."""
    n = np.asarray(rounds, float)
    return np.array([np.polyfit(n, row[side], 1)[0] for side in ("along_z_us", "equator_us")])


def life(folder: Path, exp1_folders: Sequence[Path] = ()) -> dict:
    """life-1 from its record: T1 and T2 of every qubit of the d=5 footprint, the flips per
    round of each data qubit, and each qubit's T1 and T2 three ways, the rows lifetimes.table
    prints, as rates in 1/us: IBM's (the calibration in force when life-1 ended), ours, with
    the excited share at rest π from T1's B, and in situ. In situ, each data qubit's flips a
    round read in Z, π of them |0> -> |1>; with Exp 1's jobs, both idle tests fitted together
    (in_situ); without them, the T1 of the test read in Z alone, its G holding IBM's T2 on the
    equator."""
    job = record([folder])
    after = folder / "calibration_after.json"
    claimed = Calibration.load(after) if after.exists() else job.calibration
    standard = relaxation(job.found, job.shots, claimed, job.dt)
    read_z = z_idle(job.found, job.shots, job.calibration, job.dt)
    situ = in_situ(record(exp1_folders), job) if exp1_folders else None
    alone = {row["qubit"]: row for row in read_z["qubits"]}
    joint = {row["qubit"]: row for row in situ["qubits"]} if situ else {}
    shared = ("rate_t1", "rate_t_q", "correlation", "decay_z", "decay_x", "decay_z_ibm",
              "decay_x_ibm")  # fmt: skip
    roles = role_of(place(job.calibration)[5].layout)
    rows = []
    for q in sorted(standard["T1"]):
        t1 = standard["T1"][q]
        ours = {
            f"rate_{kind.lower()}": lifetimes.rate(
                standard[kind][q]["t_us"], standard[kind][q]["sigma_us"]
            )
            for kind in lifetimes.WAITS_US
        }
        ours["excited"] = lifetimes.excited(t1["offset"], t1["sigma_offset"], claimed.readout[q])
        here = None
        if q in alone:
            flips = alone[q]["per_round"]["flips"]
            here = {
                "flips": flips,
                "up": lifetimes.product(flips, ours["excited"]),
                "rate_t1": alone[q]["rate"],
                **{key: joint[q][key] for key in shared if q in joint},
            }
        rows.append(
            {
                "qubit": q,
                "role": roles.get(q, ""),
                "ibm": {"t1_us": claimed.t1_us[q], "t2_us": claimed.t2_us[q]},
                "ours": ours,
                "in_situ": here,
            }
        )
    return {
        "folder": folder.name,
        "calibration": claimed.calibrated,
        "standard": standard,
        "z_idle": read_z,
        "in_situ": situ,
        "qubits": rows,
        "ratios": {name: (median(v) if v else None) for name, v in lifetimes.ratios(rows).items()},
    }


# Flag and relay records


def refiring(x: np.ndarray, lag: int = 1) -> float:
    """P(fires at r + lag | fired at r) - P(fires at r + lag | quiet at r), as in run 1."""
    now, later = x[:, :-lag], x[:, lag:]
    fired = (now & later).sum() / max(now.sum(), 1)
    quiet = (~now & later).sum() / max((~now).sum(), 1)
    return float(fired - quiet)


def records(
    setting: Setting,
    circuit: QuantumCircuit,
    bits: Mapping[str, np.ndarray],
    calibration: Calibration,
    dt: float,
    *,
    decay_us: Mapping[int, float] | None,
    shots: int,
    seed: int = 0,
) -> dict:
    """Each flag's and relay's values over the rounds, against the model's own shots: the
    re-firing Δ(1), the rate, and the two-state model of leakage_control (onset a, exit b,
    present or not).

    Without resets a relay's misread fires it twice in a row, its next value carrying it, so
    the model has relays re-fire (Δ(1) 0.1-0.4) and a persistent state of its own: the
    reference is the model's records, not zero. A flag's next value carries its X gauge's
    readout instead, and its Δ(1) stays near 0.
    """
    _, schedule = circuits.logical_circuit(setting)
    noisy, found, labels = memory_model(
        circuit, schedule, calibration, dt, decay_us=decay_us, flags=True
    )
    seen, _ = detectors(noisy, found, bits)
    simulated = np.asarray(noisy.compile_detector_sampler(seed=seed).sample(shots))
    columns: dict[tuple[str, int], list[int]] = defaultdict(list)
    count: dict[tuple[int, str], int] = defaultdict(int)
    for col, (r, name) in enumerate(labels):
        if " " in name:
            columns[name, count[r, name]].append(col)
            count[r, name] += 1
    rows = []
    for (name, k), cols in sorted(columns.items()):
        row: dict = {"series": name, "k": k}
        for source, data in (("chip", seen), ("model", simulated)):
            x = np.asarray(data[:, cols], dtype=bool)
            hmm = exp3.fit_hmm(x.astype(np.int8), symbols=2)
            row[source] = {
                "rate": float(x.mean()),
                "refiring": refiring(x),
                "a": hmm["a"],
                "b": hmm["b"],
                "present": hmm["present"],
            }
        rows.append(row)
    return {"setting": str(setting), "series": rows}


# Exp 1


def exp1(
    folders: Sequence[Path],
    *,
    cap: float | None = decoder.PRIMARY,
    caps: Sequence[float | str | None] = (*decoder.CAPS, decoder.CORRELATED),
    shots: int = model.SHOTS,
    seed: int = 0,
    life: Path | None = None,
) -> dict:
    """Exp 1 from its jobs: the idle test's T_q, every memory circuit decoded by each decoder,
    both fits, and the model at the measured T_q by each decoder. `cap` leads, decided before
    the run: decoder.PRIMARY, with the rule's cap (prediction.json in exp1-1's folder) and
    uniform weights beside it. With life-1's folder, also the model at the data's in-situ T1
    and T_q (in_situ), and O/M against it."""
    run = json.loads((folders[0] / "run.json").read_text())
    dt = run["dt"]
    calibration = Calibration.load(folders[0] / "calibration.json")
    found = sent(folders[0])
    data = readouts(folders)
    measured = idle(found, data, calibration, dt)
    decay = measured["decay_us"]
    caps = list(dict.fromkeys([cap, *caps]))
    rows, sent_memory = {}, {}
    for label, (setting, circuit) in found.items():
        if setting is not None and setting.decoded:
            key = setting.basis, setting.distance, setting.rounds
            rows[key] = memory(
                setting, circuit, data[label], calibration, dt, decay_us=decay, caps=caps
            )
            sent_memory[key] = (circuit, circuits.logical_circuit(setting)[1])
    count = {row["shots"] for row in rows.values()}
    if len(count) != 1:
        raise ValueError(f"memory circuits ran different shots: {sorted(count)}")
    n = count.pop()
    observed = {
        str(c): fits({key: row["failures"][str(c)] for key, row in rows.items()}, n) for c in caps
    }
    expected = model.predict(
        sent_memory, calibration, dt, decay_us=decay, shots=shots, seed=seed, decoders=caps
    )
    expected_fits = predicted_fits(expected, shots)
    flagged = [
        records(setting, circuit, data[label], calibration, dt, decay_us=decay, shots=n, seed=seed)
        for label, (setting, circuit) in found.items()
        if setting is not None and setting.decoded and setting.rounds == max(ROUNDS)
    ]
    result = {
        "folders": [folder.name for folder in folders],
        "calibration": calibration.calibrated,
        "cap": cap,
        "shots": n,
        "idle": measured,
        "observed": observed,
        "circuits": {
            str(Setting("memory", b, d, r)): row for (b, d, r), row in sorted(rows.items())
        },
        "predicted": {
            "cap": cap,
            "rule": expected["rule"],
            "shots": shots,
            "fits": expected_fits,
            "detection": _by_setting(expected["detection"]),
        },
        "observed_over_model": {
            name: against(observed[name], expected_fits[name]) for name in observed
        },
        "flags": flagged,
    }
    if life is not None:
        situ = in_situ(Record(found, data, calibration, dt), record([life]))
        again = model.predict(
            sent_memory,
            model.with_t1(calibration, situ["t1_us"]),
            dt,
            decay_us=situ["decay_us"],
            shots=shots,
            seed=seed,
            decoders=caps,
        )
        again_fits = predicted_fits(again, shots)
        result["in_situ"] = {
            "folder": life.name,
            "qubits": situ["qubits"],
            "t1_us": situ["t1_us"],
            "t_q_us": situ["decay_us"],
            "fits": again_fits,
            "detection": _by_setting(again["detection"]),
            "observed_over_model": {
                name: against(observed[name], again_fits[name]) for name in observed
            },
        }
    return result


def predicted_fits(expected: Mapping, shots: int) -> dict:
    """Both fits of the model's logical errors, by decoder, at the simulated shots."""
    return {
        name: fits({key: round(p * shots) for key, p in row["logical_error"].items()}, shots)
        for name, row in expected["decoders"].items()
    }


def _by_setting(by_key: Mapping[tuple[str, int, int], dict]) -> dict[str, dict]:
    return {str(Setting("memory", *key)): v for key, v in sorted(by_key.items())}


def against(observed: Mapping, expected: Mapping) -> dict:
    """O/M of each ε and Λ by both fits, as [value, σ], σ from both."""
    found: dict = {}
    for basis, sides in observed.items():
        ours = expected[basis]
        found[basis] = {
            fit: {
                **{
                    f"eps{d}": ratio(sides["eps"][d][fit], ours["eps"][d][fit])
                    for d in sides["eps"]
                },
                **(
                    {"lambda": ratio(sides["lambda"][fit], ours["lambda"][fit])}
                    if "lambda" in sides
                    else {}
                ),
            }
            for fit in FITS
        }
    return found


# Run 1


def run1_folders() -> list[Path]:
    """Run 1's fifteen jobs on the chip."""
    return [
        folder
        for folder in sorted(submit.RUNS.glob("ibm_fez-2026-10-0*-r*"))
        if (folder / "analysis.json").exists()
        and not json.loads((folder / "run.json").read_text()).get("offline")
    ]


def run1(
    caps: Sequence[float | None], rounds: Sequence[int] = ROUNDS, folders: Sequence[Path] = ()
) -> dict:
    """Run 1's jobs decoded with each cap's decoder and fitted on `rounds`, each job modelled
    as run 1's analysis modelled it (its calibration and its measured f); and per placement.
    Fits are keyed by str(cap)."""
    from heavyhex.experiment import run as R
    from heavyhex.experiment.circuits import SETTINGS as RUN1_SETTINGS

    jobs = []
    for folder in folders or run1_folders():
        info = json.loads((folder / "run.json").read_text())
        f = json.loads((folder / "analysis.json").read_text())["decoupled_dephasing"]
        calibration = Calibration.load(folder / "calibration.json")
        patches = [int(p) for p in R.patches_of(info)]
        shots, sent1 = R.load_shots(folder), R.load_circuits(folder)
        failures: dict[str, dict[tuple[str, int, int], int]] = defaultdict(dict)
        for i, setting in enumerate(RUN1_SETTINGS):
            if setting.kind != "memory" or setting.rounds not in rounds:
                continue
            noisy, found, _ = R.simulated(setting, sent1[i], calibration, info["dt"], f)
            fired, flips = noisy.compile_m2d_converter().convert(
                measurements=R.in_record_order(shots[i], found), separate_observables=True
            )
            dem = noisy.detector_error_model(decompose_errors=True)
            for cap in caps:
                wrong = (decoder.decode(decoder.matching(dem, cap), fired) != flips).sum(axis=0)
                for k, d in enumerate(patches):
                    failures[str(cap)][setting.basis, d, setting.rounds] = int(wrong[k])
        n = R._memory_shots(shots)
        jobs.append(
            {
                "folder": folder.name,
                "day": info["rep"]["day"],
                "placement": info["rep"]["rep"],
                "f": f,
                "shots": n,
                "fits": {
                    cap: {
                        basis: {
                            d: curve([counts[basis, d, r] for r in rounds], n, rounds)
                            for d in patches
                        }
                        for basis in "XZ"
                    }
                    for cap, counts in failures.items()
                },
            }
        )
    return {
        "rounds": list(rounds),
        "jobs": jobs,
        "placements": {str(cap): placements(jobs, str(cap)) for cap in caps},
    }


def placements(jobs: Sequence[Mapping], cap: str) -> dict:
    """Per run-1 placement, basis, distance and fit, with the cap's decoder: ε the mean over
    the days, its σ from the days' σ; Λ the inverse-variance mean of each day's ε₃/ε₅,
    widened by √(χ²/dof) above 1."""
    days: dict = defaultdict(lambda: defaultdict(dict))
    for job in jobs:
        for basis, by_d in job["fits"][cap].items():
            for d, row in by_d.items():
                days[job["placement"]][basis, d][job["day"]] = row
    found: dict = {}
    for rep, curves in sorted(days.items()):
        found[rep] = {}
        for basis in "XZ":
            found[rep][basis] = {"eps": {}, "lambda": {}}
            present = [d for d in DISTANCES if (basis, d) in curves]
            for fit in FITS:
                for d in present:
                    values = np.array([row[fit] for row in curves[basis, d].values()])
                    found[rep][basis]["eps"].setdefault(d, {})[fit] = [
                        float(values[:, 0].mean()),
                        float(sqrt((values[:, 1] ** 2).sum()) / len(values)),
                    ]
                if len(present) < 2:
                    continue
                daily = np.array(
                    [
                        ratio(curves[basis, 3][day][fit], curves[basis, 5][day][fit])
                        for day in sorted(set(curves[basis, 3]) & set(curves[basis, 5]))
                    ]
                )
                found[rep][basis]["lambda"][fit] = pooled(daily[:, 0], daily[:, 1])
    return found


def pooled(values: np.ndarray, sigmas: np.ndarray) -> list[float]:
    """The inverse-variance mean and its error, widened by √(χ²/dof) above 1."""
    w = 1 / sigmas**2
    mean = float((w * values).sum() / w.sum())
    spread = float((w * (values - mean) ** 2).sum() / max(len(values) - 1, 1))
    return [mean, sqrt(max(spread, 1.0) / w.sum())]


def against_run1(observed: Mapping, old: Mapping) -> dict:
    """ε_new/ε_run1 per basis and distance, and Λ_new - Λ_run1, against each placement of
    run 1, by both fits, as [value, σ]."""
    found: dict = {}
    for rep, bases in old.items():
        found[rep] = {}
        for basis, sides in bases.items():
            new = observed[basis]
            found[rep][basis] = {
                fit: {
                    **{
                        f"eps{d}": ratio(new["eps"][d][fit], sides["eps"][d][fit])
                        for d in DISTANCES
                    },
                    "lambda_change": [
                        new["lambda"][fit][0] - sides["lambda"][fit][0],
                        sqrt(new["lambda"][fit][1] ** 2 + sides["lambda"][fit][1] ** 2),
                    ],
                }
                for fit in FITS
            }
    return found


# Exp 3


def exp3_jobs(folders: Sequence[Path]) -> tuple[list[dict[str, np.ndarray]], float, bool]:
    """Exp 3's jobs as leakage_control.analyze takes them (each job's R0, R1 and arm), the
    cycle period in us (each arm's length over its cycles), and whether they are kerneled."""
    jobs, periods, kinds = [], set(), set()
    for folder in folders:
        run = json.loads((folder / "run.json").read_text())
        jobs.append({label: registers["m"] for label, registers in submit.load(folder).items()})
        kinds.add(run["meas_type"])
        for item in run["items"]:
            if item["label"] in exp3.ARMS:
                cycles = exp3.CYCLES * exp3.PERIODS[item["label"]]
                if item["duration_dt"] % cycles:
                    raise ValueError(f"{folder.name}: {item['label']} isn't {cycles} whole cycles")
                periods.add(item["duration_dt"] // cycles * run["dt"] * 1e6)
    if len(periods) != 1 or len(kinds) != 1:
        raise ValueError(f"the jobs differ: periods {periods}, readout {kinds}")
    return jobs, periods.pop(), kinds.pop() == exp3.MEAS_TYPE


def exp3_analysis(folders: Sequence[Path], *, reps: int = 200, seed: int = 0) -> dict:
    """Exp 3 from its jobs: the qubits re-picked from the first job's calibration."""
    jobs, period_us, kerneled = exp3_jobs(folders)
    calibration = Calibration.load(folders[0] / "calibration.json")
    selection = exp3.select(place(calibration)[3].layout)
    return exp3.analyze(jobs, selection, period_us, kerneled=kerneled, reps=reps, seed=seed)
