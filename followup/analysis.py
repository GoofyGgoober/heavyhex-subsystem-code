"""The follow-up's analysis, from the jobs' records in runs/.

Exp 1, both jobs together:

    decode   each memory circuit's detectors from its raw readouts (a value is its readout
             XOR its carry), matched on the model of the circuit as sent (program.json.gz)
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

Exp 3: each job's readouts and the cycle period, for leakage_control.analyze.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from dataclasses import dataclass
from math import exp, pi, sqrt
from typing import TYPE_CHECKING

import numpy as np

from heavyhex.patches.placement import Calibration
from heavyhex.simulation.noisy import fit_per_round

from . import circuits, decoder, model, submit
from . import leakage_control as exp3
from .circuits import DISTANCES, ROUNDS, Setting
from .placement import place

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    import stim
    from qiskit import QuantumCircuit

MIN_R = 0.05  # an idle length whose r is below this says nothing about T_q
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


def equator_times(circuit: QuantumCircuit, qubits: Sequence[int], dt: float) -> dict[int, float]:
    """Each qubit's delay time in us with its state on the equator, up to its readout.

    In the idle test a data qubit gets one-qubit gates only, so its state stays along X, Y or
    Z. Each delay takes e^(-t/T) of its ⟨X⟩ on the equator, and e^(-t/T1) along Z, between a
    wait's H gates: only the equator's time measures T_q.
    """
    swap = {"rz": {"X": "Y", "Y": "X"}, "sx": {"Y": "Z", "Z": "Y"}}
    axis = dict.fromkeys(qubits, "Z")
    total = dict.fromkeys(qubits, 0.0)
    for item in circuit.data:
        op = item.operation
        for q in (circuit.find_bit(bit).index for bit in item.qubits):
            if axis.get(q, "read") == "read":
                continue
            if op.name == "measure":
                axis[q] = "read"
            elif op.name == "delay" and axis[q] != "Z":
                total[q] += model._ns(op, dt) / 1e3
            elif op.name == "sx" or (op.name == "rz" and round(float(op.params[0]) / (pi / 2)) % 2):
                axis[q] = swap[op.name].get(axis[q], axis[q])
            elif op.name not in ("rz", "x", "delay", "barrier"):
                raise ValueError(f"{op.name} on qubit {q}: the idle test has one-qubit gates only")
    return total


def data_qubits(circuit: QuantumCircuit) -> dict[int, int]:
    """The chip qubit read into each bit of the data register."""
    found = {}
    for item in circuit.data:
        if item.operation.name == "measure":
            register, i = circuit.find_bit(item.clbits[0]).registers[0]
            if register.name.startswith("d"):
                found[i] = circuit.find_bit(item.qubits[0]).index
    return dict(sorted(found.items()))


def pulses(circuit: QuantumCircuit, calibration: Calibration, dt: float) -> np.ndarray:
    """G: what the model leaves of each data qubit's ⟨X⟩ but for its decay on the equator, by
    data bit. The product of 1 - 2p over the errors that flip its readout, times e^(t/T) for
    the e^(-t/T) its time t on the equator takes (T = IBM's T2, at most 2 T1, as the model
    has it)."""
    import stim

    noisy, records = model.noisy(circuit, calibration, dt)
    read = data_qubits(circuit)
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
    times = equator_times(circuit, list(read.values()), dt)
    for k, q in enumerate(read.values()):
        x[k] *= exp(times[q] / min(calibration.t2_us[q], 2 * calibration.t1_us[q]))
    return x


def decay_time(
    t: np.ndarray, x: np.ndarray, y: np.ndarray, nx: np.ndarray, ny: np.ndarray, g: np.ndarray
) -> dict:
    """T_q and its error from one qubit's ⟨X⟩ and ⟨Y⟩ over the idle lengths: ln(r/G) = c - t/T_q,
    weighted, r² less the shot noise in x² and y². NaN with fewer than two r above MIN_R."""
    r = np.sqrt(np.maximum(x**2 + y**2 - (1 - x**2) / nx - (1 - y**2) / ny, 0))
    keep = r > MIN_R
    nan = float("nan")
    if keep.sum() < 2:
        return {
            "t_q_us": nan,
            "sigma_us": nan,
            "offset": nan,
            "chi2_red": nan,
            "used": int(keep.sum()),
        }
    var = (x**2 * (1 - x**2) / nx + y**2 * (1 - y**2) / ny)[keep] / r[keep] ** 4
    t, ln = t[keep], np.log(r[keep] / g[keep])
    (b, c), cov = np.polyfit(t, ln, 1, w=1 / np.sqrt(var), cov="unscaled")
    chi2 = float((((ln - c - b * t) ** 2) / var).sum())
    return {
        "t_q_us": float(-1 / b) if b < 0 else float("inf"),
        "sigma_us": float(sqrt(cov[0, 0]) / b**2),
        "offset": float(exp(c)),
        "chi2_red": chi2 / (keep.sum() - 2) if keep.sum() > 2 else nan,
        "used": int(keep.sum()),
    }


def idle(
    found: Mapping[str, tuple[Setting | None, QuantumCircuit]],
    shots: Mapping[str, Mapping[str, np.ndarray]],
    calibration: Calibration,
    dt: float,
) -> dict:
    """Each data qubit's T_q from the idle test, and the decay times the model takes:
    the measured T_q where finite, else IBM's T2."""
    runs: dict[str, dict[int, str]] = {"X": {}, "Y": {}}
    for label, (setting, _) in found.items():
        if setting is not None and setting.kind == "idle":
            runs[setting.readout][setting.rounds] = label
    rounds = sorted(runs["X"])
    if sorted(runs["Y"]) != rounds:
        raise ValueError("the idle test needs the same lengths read in X and in Y")
    first = found[runs["X"][rounds[0]]][1]
    read = data_qubits(first)
    register = next(name for name in shots[runs["X"][rounds[0]]] if name.startswith("d"))
    t, g = [], []
    for n in rounds:
        circuit = found[runs["X"][n]][1]
        times = equator_times(circuit, list(read.values()), dt)
        t.append([times[q] for q in read.values()])
        g.append(pulses(circuit, calibration, dt))
    t_us, g_all = np.array(t), np.array(g)
    x, y, nx, ny = ([] for _ in range(4))
    for n in rounds:
        bx, by = shots[runs["X"][n]][register], shots[runs["Y"][n]][register]
        x.append(1 - 2 * bx.mean(axis=0))
        y.append(1 - 2 * by.mean(axis=0))
        nx.append(len(bx))
        ny.append(len(by))
    x_all, y_all = np.array(x), np.array(y)
    rows = []
    for k, (i, q) in enumerate(read.items()):
        n_x, n_y = np.array(nx, float), np.array(ny, float)
        fit = decay_time(t_us[:, k], x_all[:, k], y_all[:, k], n_x, n_y, g_all[:, k])
        rows.append(
            {
                "data": i,
                "qubit": q,
                "t1_us": calibration.t1_us[q],
                "t2_us": calibration.t2_us[q],
                **fit,
                "equator_us": t_us[:, k].tolist(),
                "pulses": g_all[:, k].tolist(),
                "x": x_all[:, k].tolist(),
                "y": y_all[:, k].tolist(),
            }
        )
    decay = {  # no decay seen: inf, which the model caps at 2 T1
        row["qubit"]: calibration.t2_us[row["qubit"]] if np.isnan(row["t_q_us"]) else row["t_q_us"]
        for row in rows
    }
    return {"rounds": rounds, "qubits": rows, "decay_us": decay}


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
    cap: float | None,
    caps: Sequence[float | str | None] = (*decoder.CAPS, decoder.CORRELATED),
    shots: int = model.SHOTS,
    seed: int = 0,
) -> dict:
    """Exp 1 from its jobs: the idle test's T_q, every memory circuit decoded by each decoder,
    both fits, and the model at the measured T_q by each decoder. `cap` leads: the rule's,
    from the prediction saved before the run (prediction.json in exp1-1's folder)."""
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
    expected_fits = {
        name: fits({key: round(p * shots) for key, p in row["logical_error"].items()}, shots)
        for name, row in expected["decoders"].items()
    }
    flagged = [
        records(setting, circuit, data[label], calibration, dt, decay_us=decay, shots=n, seed=seed)
        for label, (setting, circuit) in found.items()
        if setting is not None and setting.decoded and setting.rounds == max(ROUNDS)
    ]
    return {
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
            "rule": expected["cap"],
            "shots": shots,
            "fits": expected_fits,
            "detection": {
                str(Setting("memory", *key)): v for key, v in sorted(expected["detection"].items())
            },
        },
        "observed_over_model": {
            name: against(observed[name], expected_fits[name]) for name in observed
        },
        "flags": flagged,
    }


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
