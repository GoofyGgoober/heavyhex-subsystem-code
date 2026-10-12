"""Rerun the prespecified analysis (run.analyze, then run.combine) with the corrected XY variance.

Post hoc, exploratory; no QPU. Each job folder's inputs are copied to a scratch runs
directory (argument 1), run._idle_dephasing is replaced by a copy whose only change is the
weight of each ln r point (delta-method variance for separate <X>, <Y> samples, see
xy_fix.py), and run.analyze / run.combine are called unchanged on the copies. Jobs named
after argument 1 are reanalyzed; every other job keeps its committed analysis.json.
The new analysis.json files and combined.json stay in the scratch directory; a summary is
written to reanalysis.json here.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from math import sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from heavyhex.experiment import run as R  # noqa: E402
from heavyhex.experiment.circuits import SETTINGS  # noqa: E402
from xy_fix import fit, job_folders  # noqa: E402

INPUTS = ("run.json", "calibration.json", "calibration_after.json", "circuits.qpy.gz",
          "job.json", "prediction.json", "shots.npz")


def corrected_idle_dephasing(shots, circuits, calibration, dt):
    """run._idle_dephasing (run.py:921-972) with the corrected weight; nothing else changed."""
    idle = [i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.readout == "X"]
    in_y = {s.rounds: i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.readout == "Y"}
    rounds = np.array([SETTINGS[i].rounds for i in idle])
    xs = [R.x_values(shots[i]) for i in idle]
    ys = [R.x_values(shots[in_y[n]]) for n in rounds]
    ends = [[R._exact_x(circuits[i], calibration, dt, f) for i in idle] for f in R.ENDS]
    found = {key: {} for key in ("per_qubit", "uncertainty", "x_only", "phase_per_round")}
    for d in xs[0]:
        for found_d in found.values():
            found_d[d] = []
        for q in range(int(d) ** 2):
            x = np.array([v[d][q] for v in xs])
            y = np.array([v[d][q] for v in ys])
            count_x = np.array([len(shots[i][f"d{d}"]) for i in idle])
            count_y = np.array([len(shots[in_y[n]][f"d{d}"]) for n in rounds])
            r2 = x**2 + y**2 - (1 - x**2) / count_x - (1 - y**2) / count_y
            r = np.sqrt(np.maximum(r2, 0))
            log_x0, log_x1 = (np.log([e[d][q] for e in end]) for end in ends)
            ax, ay = 1 + 1 / count_x, 1 + 1 / count_y
            var_r2 = 4 * (ax**2 * x**2 * (1 - x**2) / count_x + ay**2 * y**2 * (1 - y**2) / count_y)
            with np.errstate(divide="ignore", invalid="ignore"):
                var_log = var_r2 / (4 * r2**2)
            share, sigma = fit(r, log_x0, log_x1, var_log)
            found["per_qubit"][d].append(share)
            found["uncertainty"][d].append(sigma)
            found["x_only"][d].append(R._fit_share(x, log_x0, log_x1, count_x)[0])
            found["phase_per_round"][d].append(R._phase(rounds, x, y, r))
    pairs = [
        (s, u)
        for d in found["per_qubit"]
        for s, u in zip(found["per_qubit"][d], found["uncertainty"][d])
        if np.isfinite(s) and np.isfinite(u)
    ]
    every = [s for s, _ in pairs]
    x_only = [s for d in found["x_only"] for s in found["x_only"][d] if np.isfinite(s)]
    return {
        **found,
        "uniformity": R._uniformity(np.array(every), np.array([u for _, u in pairs])),
        "median": float(np.median(every)),
        "median_uncertainty": float(1.25 * np.std(every) / sqrt(len(every))),
        "x_only_median": float(np.median(x_only)),
    }


def main() -> None:
    scratch = Path(sys.argv[1])
    redo = set(sys.argv[2:])
    runs = scratch / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for folder in job_folders():
        dest = runs / folder.name
        dest.mkdir(exist_ok=True)
        for name in INPUTS:
            if (folder / name).exists() and not (dest / name).exists():
                shutil.copy2(folder / name, dest / name)
        if folder.name not in redo and not (dest / "analysis.json").exists():
            shutil.copy2(folder / "analysis.json", dest / "analysis.json")
    control = os.environ.get("XY_CONTROL") == "1"  # unpatched rerun, same environment
    if not control:
        R._idle_dephasing = corrected_idle_dephasing
    R.RUNS = runs
    for name in sorted(redo):
        print("analyzing", name, flush=True)
        R.analyze(runs / name)
    old = json.loads((REPO / "runs" / "combined.json").read_text())
    new = R.combine()
    out = {"reanalyzed": sorted(redo), "model": {}, "pooled_x": {}, "reps": []}
    for test in new["model"]:
        out["model"][test] = [old["model"][test]["chi2"], new["model"][test]["chi2"],
                              old["model"][test]["agrees"], new["model"][test]["agrees"]]
    for key in ("lambda", "lambda_uncertainty", "predicted", "verdict"):
        out["pooled_x"][key] = [old["pooled_x"][key], new["pooled_x"][key]]
    for a, b in zip(old["reps"], new["reps"]):
        row = {"day": a["day"], "rep": a["rep"], "verdict": [a["verdict"], b["verdict"]], "tests": {}}
        for test in a["tests"]:
            ta, tb = a["tests"][test], b["tests"][test]
            row["tests"][test] = {k: [ta[k], tb[k]] for k in ("observed", "predicted", "bound", "agrees")}
        out["reps"].append(row)
    for name in sorted(redo):
        a = json.loads((REPO / "runs" / name / "analysis.json").read_text())
        b = json.loads((runs / name / "analysis.json").read_text())
        out.setdefault("jobs", {})[name] = {
            "f": [a["decoupled_dephasing"], b["decoupled_dephasing"]],
            "median_uncertainty": [a["idle_test"]["median_uncertainty"], b["idle_test"]["median_uncertainty"]],
            "dephasing_from_memory": [a["dephasing_from_memory"], b["dephasing_from_memory"]],
            "agreement": {t: {k: [a["agreement"]["tests"][t][k], b["agreement"]["tests"][t][k]]
                              for k in ("observed", "predicted", "from_f", "bound", "agrees")}
                          for t in a["agreement"]["tests"]},
            "per_qubit_prediction": {
                basis: [a["predicted_with_each_qubits_dephasing"]["fits"].get(basis),
                        b["predicted_with_each_qubits_dephasing"]["fits"].get(basis)]
                for basis in "XZ"
            },
        }
    (HERE / ("reanalysis_control.json" if control else "reanalysis.json")).write_text(json.dumps(out, indent=1, default=str))
    print(json.dumps({k: out[k] for k in ("model", "pooled_x")}, indent=1))


if __name__ == "__main__":
    main()
