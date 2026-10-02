"""Prepare, submit and analyze a run. Each run keeps everything in its own folder.

prepare: place the patches from the day's calibration, translate the circuits
for the chip, and freeze the simulator's prediction. rehearse: fake the chip's
shots with the simulator. submit: send the circuits (uses the QPU). analyze:
decode the shots and compare with the prediction.
"""

from __future__ import annotations

import gzip
import json
from dataclasses import asdict
from datetime import datetime
from math import log, sqrt
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ..circuits.flagged import memory_circuit_flagged
from ..patches.operators import build_operators
from ..patches.placement import Calibration, fez_qubits
from ..simulation.noisy import add_detectors, fit_per_round
from ..simulation.scheduled import noisy_scheduled
from .circuits import DISTANCES, IDLE_ROUNDS, ROUNDS, SETTINGS, Setting, for_fez, logical_circuit

if TYPE_CHECKING:
    import numpy as np
    import stim
    from qiskit import QuantumCircuit

RUNS = Path(__file__).resolve().parents[3] / "runs"
SIMULATED_SHOTS = 50_000
ENDS = (0.0, 1.0)  # decoupled_dephasing at the two ends of the prediction


def prepare(folder: Path, calibration: Calibration, target: Any, *, offline: bool = False) -> dict:
    """Write the run's calibration, circuits and prediction into a new folder."""
    from qiskit import qpy

    chips = {d: fez_qubits(d, calibration) for d in DISTANCES}
    circuits = [for_fez(logical_circuit(s, chips, target.num_qubits)[0], target) for s in SETTINGS]
    folder.mkdir(parents=True)
    calibration.save(folder / "calibration.json")
    with gzip.open(folder / "circuits.qpy.gz", "wb") as f:
        qpy.dump(circuits, f)
    run = {
        "backend": calibration.backend,
        "dt": target.dt,
        "offline": offline,
        "prepared": _now(),
        "chips": {str(d): chip for d, chip in chips.items()},
        "settings": [asdict(s) for s in SETTINGS],
        "seconds": [c.estimate_duration(target, unit="s") for c in circuits],
    }
    _write(folder / "run.json", run)
    prediction = {
        str(f): predict(circuits, calibration, target.dt, decoupled_dephasing=f) for f in ENDS
    }
    _write(folder / "prediction.json", prediction)
    return prediction


def predict(
    circuits: list[QuantumCircuit],
    calibration: Calibration,
    dt: float,
    *,
    decoupled_dephasing: float,
) -> dict:
    """What the simulator expects of each circuit, then the fits."""
    settings = []
    for setting, circuit in zip(SETTINGS, circuits):
        model, records, labels = simulated(setting, circuit, calibration, dt, decoupled_dephasing)
        if setting.kind == "memory":
            sampler = model.compile_detector_sampler(seed=setting.rounds)
            detectors, flips = sampler.sample(SIMULATED_SHOTS, separate_observables=True)
            failed = _decode(model, detectors) != flips
            rates = detectors.mean(axis=0).tolist()
            settings.append(
                {
                    "logical_error": failed.mean(axis=0).tolist(),
                    "detector_rates": rates,
                    "detectors": labels,
                    "by_round": _by_round(labels, rates),
                }
            )
        else:
            measured = model.compile_sampler(seed=setting.rounds).sample(SIMULATED_SHOTS)
            settings.append({"x": x_values(by_register(measured, records))})
    return {"settings": settings, **_fits(settings, SIMULATED_SHOTS)}


def simulated(
    setting: Setting,
    circuit: QuantumCircuit,
    calibration: Calibration,
    dt: float,
    decoupled_dephasing: float,
) -> tuple[stim.Circuit, dict[tuple[str, int], int], list]:
    """The stim circuit for a translated circuit; memory gets detectors and one logical per patch.

    Returns the circuit, the measurement of each (register, bit), and detector labels
    as (distance, round, stabilizer).
    """
    model, records = noisy_scheduled(
        circuit, calibration, dt, decoupled_dephasing=decoupled_dephasing
    )
    labels: list = []
    if setting.kind == "memory":
        for k, d in enumerate(DISTANCES):
            _, schedule = memory_circuit_flagged(
                build_operators(d), rounds=setting.rounds, basis=setting.basis
            )
            mine = {(reg[0], i): m for (reg, i), m in records.items() if reg[1:] == str(d)}
            found: list = []
            add_detectors(model, schedule, mine, observable=k, labels=found)
            labels += [(d, *label) for label in found]
    return model, records, labels


def submit(folder: Path, backend: Any, shots: int) -> None:
    """Send the run's circuits and save every shot. Uses QPU time on a real backend."""
    from qiskit_ibm_runtime import SamplerV2

    run = _read(folder / "run.json")
    if run["offline"] and not _is_local(backend):
        raise ValueError("this run was prepared offline; prepare it again before using the QPU")
    sampler = SamplerV2(mode=backend)
    sampler.options.default_shots = shots
    if not _is_local(backend):
        # IBM's own decoupling and twirling would change the circuit the prediction is for.
        sampler.options.dynamical_decoupling.enable = False
        sampler.options.twirling.enable_gates = False
        sampler.options.twirling.enable_measure = False
    job = sampler.run(load_circuits(folder))
    _write(
        folder / "job.json",
        {"job": job.job_id(), "backend": backend.name, "shots": shots, "submitted": _now()},
    )
    save_shots(folder, job.result())


def rehearse(folder: Path, shots: int, decoupled_dephasing: float, seed: int = 0) -> None:
    """Write simulated shots in place of the chip's, to try out the analysis."""
    import numpy as np

    run = _read(folder / "run.json")
    calibration = Calibration.load(folder / "calibration.json")
    saved = {}
    for i, (setting, circuit) in enumerate(zip(SETTINGS, load_circuits(folder))):
        model, records, _ = simulated(setting, circuit, calibration, run["dt"], decoupled_dephasing)
        measured = model.compile_sampler(seed=seed + i).sample(shots)
        for name, bits in by_register(measured, records).items():
            saved[f"{i}/{name}"] = bits
    np.savez_compressed(folder / "shots.npz", **saved)


def fetch(folder: Path) -> None:
    """Get the shots of a submitted job from IBM (read-only)."""
    from qiskit_ibm_runtime import QiskitRuntimeService

    job = QiskitRuntimeService().job(_read(folder / "job.json")["job"])
    save_shots(folder, job.result())


def save_shots(folder: Path, result: Any) -> None:
    import numpy as np

    shots = {
        f"{i}/{name}": bits.to_bool_array(order="little")
        for i, pub in enumerate(result)
        for name, bits in pub.data.items()
    }
    np.savez_compressed(folder / "shots.npz", **shots)


def analyze(folder: Path, decoupled_dephasing: float | None = None) -> dict:
    """Decode the shots and set them against the prediction.

    The idle test gives decoupled_dephasing, unless one is given; the memory
    shots are decoded, and the simulator rerun, with that value.
    """
    import numpy as np

    run = _read(folder / "run.json")
    calibration = Calibration.load(folder / "calibration.json")
    prediction = _read(folder / "prediction.json")
    circuits = load_circuits(folder)
    shots = load_shots(folder)
    count = next(iter(shots[0].values())).shape[0]

    measured_f = _idle_dephasing(shots, prediction)
    if decoupled_dephasing is None and not np.isfinite(measured_f["median"]):
        raise ValueError("the idle test gave no dephasing; pass one to analyze")
    f = measured_f["median"] if decoupled_dephasing is None else decoupled_dephasing
    f = max(float(f), 0.0)  # above 1 is physical (more noise than the echo T2), below 0 is not
    observed = []
    for i, (setting, circuit) in enumerate(zip(SETTINGS, circuits)):
        model, records, labels = simulated(setting, circuit, calibration, run["dt"], f)
        if setting.kind == "memory":
            measurements = in_record_order(shots[i], records)
            detectors, flips = model.compile_m2d_converter().convert(
                measurements=measurements, separate_observables=True
            )
            failed = _decode(model, detectors) != flips
            rates = detectors.mean(axis=0).tolist()
            observed.append(
                {
                    "logical_error": failed.mean(axis=0).tolist(),
                    "detector_rates": rates,
                    "detectors": labels,
                    "by_round": _by_round(labels, rates),
                }
            )
        else:
            observed.append({"x": x_values(shots[i])})
    expected = predict(circuits, calibration, run["dt"], decoupled_dephasing=f)
    count_detectors = sum(len(s["detector_rates"]) for s in observed if "detector_rates" in s)
    analysis = {
        "decoupled_dephasing": f,
        "chance_excess": sqrt(2 * log(max(count_detectors, 2))),
        "idle_test": measured_f,
        "observed": {"settings": observed, **_fits(observed, count)},
        "predicted_at_measured_dephasing": expected,
        "worst_detectors": _worst_detectors(observed, expected["settings"], count),
    }
    _write(folder / "analysis.json", analysis)
    return analysis


def load_circuits(folder: Path) -> list[QuantumCircuit]:
    from qiskit import qpy

    with gzip.open(folder / "circuits.qpy.gz", "rb") as f:
        return qpy.load(f)


def load_shots(folder: Path) -> list[dict[str, np.ndarray]]:
    """Per circuit, each register's bits as shots x bits."""
    import numpy as np

    saved = np.load(folder / "shots.npz")
    shots: list[dict[str, np.ndarray]] = [{} for _ in SETTINGS]
    for key in saved.files:
        i, name = key.split("/")
        shots[int(i)][name] = saved[key]
    return shots


def default_folder(backend: str, offline: bool) -> Path:
    return RUNS / f"{backend}-{datetime.now():%Y-%m-%d}{'-offline' if offline else ''}"


def _decode(model: stim.Circuit, detectors: np.ndarray) -> np.ndarray:
    import pymatching

    matching = pymatching.Matching.from_detector_error_model(
        model.detector_error_model(decompose_errors=True)
    )
    return matching.decode_batch(detectors).astype(bool)


def by_register(measured: np.ndarray, records: dict) -> dict[str, np.ndarray]:
    """Simulated shots (shots x measurements) as the chip returns them: per register."""
    import numpy as np

    sizes: dict[str, int] = {}
    for name, i in records:
        sizes[name] = max(sizes.get(name, 0), i + 1)
    bits = {name: np.zeros((len(measured), size), dtype=bool) for name, size in sizes.items()}
    for (name, i), k in records.items():
        bits[name][:, i] = measured[:, k]
    return bits


def in_record_order(bits: dict[str, np.ndarray], records: dict) -> np.ndarray:
    import numpy as np

    count = next(iter(bits.values())).shape[0]
    out = np.zeros((count, len(records)), dtype=bool)
    for (name, i), k in records.items():
        out[:, k] = bits[name][:, i]
    return out


def x_values(bits: dict[str, np.ndarray]) -> dict[str, list[float]]:
    """<X> of each data qubit, per patch, from the final readout."""
    return {
        str(d): [float(1 - 2 * bits[f"d{d}"][:, q].mean()) for q in range(d * d)] for d in DISTANCES
    }


def _decay_rates(x_by_rounds: list[dict[str, list[float]]]) -> dict[str, list[float]]:
    """How fast each data qubit's <X> falls per round of the idle test (log slope)."""
    import numpy as np

    rates = {}
    for d in map(str, DISTANCES):
        values = np.array([x[d] for x in x_by_rounds])  # rounds x qubits
        rates[d] = []
        for column in values.T:
            keep = column > 0.05
            if keep.sum() < 2:
                rates[d].append(float("nan"))
                continue
            slope, _ = np.polyfit(np.array(IDLE_ROUNDS)[keep], np.log(column[keep]), 1)
            rates[d].append(float(-slope))
    return rates


def _idle_dephasing(shots: list, prediction: dict) -> dict:
    """Place each data qubit's measured decay between the two predicted ends."""
    import numpy as np

    idle = [i for i, s in enumerate(SETTINGS) if s.kind == "idle"]
    measured = _decay_rates([x_values(shots[i]) for i in idle])
    ends = [_decay_rates([prediction[str(f)]["settings"][i]["x"] for i in idle]) for f in ENDS]
    share = {
        d: [
            float((m - low) / (high - low)) if high > low else float("nan")
            for m, low, high in zip(measured[d], ends[0][d], ends[1][d])
        ]
        for d in measured
    }
    every = [s for d in share for s in share[d] if np.isfinite(s)]
    return {
        "per_qubit": share,
        "median": float(np.median(every)) if every else float("nan"),
    }


def _by_round(labels: list, rates: list) -> dict[str, list[float]]:
    """Mean firing rate of each patch's stabilizer detectors, round by round.

    A rise over the rounds that the prediction lacks points to leakage or heating.
    """
    sums: dict[tuple[int, int], list[float]] = {}
    for (d, r, name), rate in zip(labels, rates):
        if " " not in name:  # stabilizers, not flags or relays
            sums.setdefault((d, r), []).append(rate)
    return {
        str(d): [sum(v) / len(v) for (dd, _), v in sorted(sums.items()) if dd == d]
        for d in DISTANCES
    }


def _worst_detectors(observed: list, expected: list, shots: int, many: int = 10) -> list:
    """The detectors that fire most above the prediction, in standard errors.

    With N detectors in all, chance alone reaches about sqrt(2 ln N) standard errors.
    """
    import numpy as np

    rows = []
    for setting, seen, sim in zip(SETTINGS, observed, expected):
        if setting.kind != "memory":
            continue
        got, want = np.array(seen["detector_rates"]), np.array(sim["detector_rates"])
        excess = (got - want) / np.sqrt(np.maximum(want * (1 - want), 1e-6) / shots)
        for k in np.argsort(-excess)[:many]:
            rows.append(
                {
                    "setting": str(setting),
                    "detector": seen["detectors"][k],
                    "observed": float(got[k]),
                    "predicted": float(want[k]),
                    "excess": float(excess[k]),
                }
            )
    return sorted(rows, key=lambda r: -r["excess"])[:many]


def _fits(settings: list[dict], shots: int) -> dict:
    """Error per round per basis and patch, and Λ."""
    fits: dict = {}
    for basis in "XZ":
        rows = [r for s, r in zip(SETTINGS, settings) if s.kind == "memory" and s.basis == basis]
        per_round = {}
        for k, d in enumerate(DISTANCES):
            fit = fit_per_round(ROUNDS, [r["logical_error"][k] for r in rows], shots)
            per_round[str(d)] = [fit.per_round, fit.uncertainty]
        (e3, u3), (e5, u5) = per_round["3"], per_round["5"]
        lam = e3 / e5
        sigma = lam * sqrt((u3 / e3) ** 2 + (u5 / e5) ** 2)  # one standard error
        verdict = "undecided"
        if lam - 1.645 * sigma > 1:
            verdict = "d = 5 better"
        elif lam + 1.645 * sigma < 1:
            verdict = "d = 5 worse"
        fits[basis] = {
            "per_round": per_round,
            "lambda": lam,
            "lambda_uncertainty": sigma,
            "verdict": verdict,  # one-sided, 95%
        }
    return {"fits": fits}


def _is_local(backend: Any) -> bool:
    return type(backend).__module__.startswith("qiskit_aer")


def _now() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _read(path: Path) -> dict:
    return json.loads(path.read_text())


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=1) + "\n")
