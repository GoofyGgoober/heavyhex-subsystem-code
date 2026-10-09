"""What did the resets cost? The calibration model with every reset taking no time.

Each round resets every ancilla, flag and relay before use: two resets of 1.68 µs in a
7.9 µs round, while the data qubits wait. A measured qubit is already in a known state,
so the resets could have been replaced by tracking its outcome in software. This gives
every reset zero duration and keeps its error, a stand-in for that: the data wait less,
and a reused qubit still starts wrong as often as a reset would leave it wrong.

Two settings, X and Z memory, as run and without reset time:
  uniform    the chip at the last run's medians, as uniform_scale.py at scale 1 (f = 1)
  placement  run 2026-10-04-r1 with its own calibration and measured f, its circuits
             translated again for Fez with the calibration's median gate times

    python docs/diagnosis/no_reset.py [shots]   # writes docs/diagnosis/results/no_reset.json
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import cache
from math import sqrt

from common import REPO, jobs
from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import ROUNDS, SETTINGS, for_fez, logical_circuit, with_durations
from heavyhex.patches.placement import Calibration
from heavyhex.simulation.noisy import decay, fit_per_round, typical

FOLDER = REPO / "runs/ibm_fez-2026-10-04-r1"
MEMORY = [k for k, s in enumerate(SETTINGS) if s.kind == "memory"]


def without_reset_time(calibration: Calibration) -> Calibration:
    return replace(calibration, durations_ns={**calibration.durations_ns, "reset": 0})


def uniform(task: tuple) -> tuple:
    reset_time, basis, d, shots = task
    cal = typical(jobs()[-1].calibration)
    if not reset_time:
        cal = without_reset_time(cal)
    fit = decay(d, basis, cal, shots=shots, seed=5, decoupling=True, decoupled_dephasing=1.0)
    return task, (fit.per_round, fit.uncertainty)


@cache
def translated(reset_time: bool) -> tuple[list, list]:
    """The run's circuits translated again, and each one's length in seconds."""
    from qiskit_ibm_runtime.fake_provider import FakeFez

    info = json.loads((FOLDER / "run.json").read_text())
    cal = Calibration.load(FOLDER / "calibration.json")
    if not reset_time:
        cal = without_reset_time(cal)
    target = with_durations(FakeFez().target, cal.durations_ns)
    circuits = [for_fez(logical_circuit(s, info["chips"], target.num_qubits)[0], target) for s in SETTINGS]
    return circuits, [c.estimate_duration(target, unit="s") for c in circuits]


def placement(task: tuple) -> tuple:
    reset_time, k, shots = task
    info = json.loads((FOLDER / "run.json").read_text())
    f = json.loads((FOLDER / "analysis.json").read_text())["decoupled_dephasing"]
    cal = Calibration.load(FOLDER / "calibration.json")
    model, _, _ = R.simulated(SETTINGS[k], translated(reset_time)[0][k], cal, info["dt"], f)
    detectors, flips = model.compile_detector_sampler(seed=SETTINGS[k].rounds).sample(shots, separate_observables=True)
    return task, (R._decode(model, detectors) != flips).mean(axis=0).tolist()


def row(e3: tuple, e5: tuple) -> dict:
    lam = e3[0] / e5[0]
    return {"eps3": e3[0], "eps5": e5[0], "lambda": lam,
            "lambda_unc": lam * sqrt((e3[1] / e3[0]) ** 2 + (e5[1] / e5[0]) ** 2)}


def main() -> None:
    shots = int(sys.argv[1]) if len(sys.argv) > 1 else 200000
    out = {"shots_per_circuit": shots, "uniform": {}, "placement": {"folder": FOLDER.name}}
    with ProcessPoolExecutor(4) as pool:
        tasks = [(t, b, d, shots) for t in (True, False) for b in "XZ" for d in (3, 5)]
        fits = dict(pool.map(uniform, tasks))
        tasks = [(t, k, shots) for t in (True, False) for k in MEMORY]
        failed = dict(pool.map(placement, tasks))
    for t, name in ((True, "as_run"), (False, "no_reset_time")):
        out["uniform"][name] = {b: row(fits[(t, b, 3, shots)], fits[(t, b, 5, shots)]) for b in "XZ"}
        seconds = translated(t)[1]
        x = [k for k in MEMORY if SETTINGS[k].basis == "X"]
        result = {"round_us": 1e6 * (seconds[x[1]] - seconds[x[0]])}
        for b in "XZ":
            rows = [k for k in MEMORY if SETTINGS[k].basis == b]
            e = {}
            for i, d in enumerate((3, 5)):
                fit = fit_per_round(ROUNDS, [failed[(t, k, shots)][i] for k in rows], shots)
                e[d] = (fit.per_round, fit.uncertainty)
            result[b] = row(e[3], e[5])
        out["placement"][name] = result
    for setting in ("uniform", "placement"):
        for name in ("as_run", "no_reset_time"):
            for b in "XZ":
                r = out[setting][name][b]
                print(f"{setting:9s} {name:13s} {b}: eps3 {100*r['eps3']:.2f}%  eps5 {100*r['eps5']:.2f}%  "
                      f"Λ {r['lambda']:.3f} ± {r['lambda_unc']:.3f}")
    (REPO / "docs/diagnosis/results/no_reset.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
