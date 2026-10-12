"""The prespecified analysis rerun (run.analyze, then run.combine) with the corrected idle-test
variance (idle_f.py), and an unpatched control rerun in the same environment.

`run SCRATCH JOB...` copies each job folder's inputs to SCRATCH/runs, replaces
run._idle_dephasing by a copy whose only change is the weight of each ln r point, and calls
run.analyze and run.combine unchanged on the copies; the jobs not named keep their committed
analysis.json. With --control nothing is replaced. It writes
results/idle_f_reanalysis[_control].json (old and new side by side) and
results/idle_f_combined_{fixed,control}.json (the rerun's combined.json).

`compare CONTROL FIXED` sets the two reruns against each other (both runs directories), so
that the fix's effect is not mixed with Monte Carlo differences of the prediction between
environments: results/idle_f_compare.json.

Heavy: every job re-analysed, 1-2 GB of memory each; src/ and runs/ are not touched.

    python docs/analysis/idle_f_reanalyze.py run /tmp/xy ibm_fez-2026-10-04-r1 ...           # corrected
    python docs/analysis/idle_f_reanalyze.py run /tmp/xy_control --control ibm_fez-2026-10-04-r1 ...
    python docs/analysis/idle_f_reanalyze.py compare /tmp/xy_control/runs /tmp/xy/runs
"""

from __future__ import annotations

import argparse
import json
import shutil
from math import sqrt
from pathlib import Path

import numpy as np

from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import SETTINGS

import perqubit_f as PQ
from common import RESULTS, RUNS, job_folders
from idle_f import fit

INPUTS = (
    "run.json",
    "calibration.json",
    "calibration_after.json",
    "circuits.qpy.gz",
    "job.json",
    "prediction.json",
    "shots.npz",
)
TESTS = ("X lambda", "X eps3", "X eps5", "Z eps3", "Z eps5", "Z lambda")


def corrected_idle_dephasing(shots, circuits, calibration, dt):
    """run._idle_dephasing (run.py:921-972) with the corrected weight; nothing else changed."""
    idle = [i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.readout == "X"]
    in_y = {s.rounds: i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.readout == "Y"}
    rounds = np.array([SETTINGS[i].rounds for i in idle])
    xs = [R.x_values(shots[i]) for i in idle]
    ys = [R.x_values(shots[in_y[n]]) for n in rounds]
    ends = [[R._exact_x(circuits[i], calibration, dt, f) for i in idle] for f in R.ENDS]
    found: dict = {key: {} for key in ("per_qubit", "uncertainty", "x_only", "phase_per_round")}
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


def rerun(scratch: Path, redo: set[str], control: bool) -> None:
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
    if not control:
        R._idle_dephasing = corrected_idle_dephasing
    R.RUNS = runs
    for name in sorted(redo):
        print("analyzing", name, flush=True)
        R.analyze(runs / name)
    old = json.loads((RUNS / "combined.json").read_text())
    new = R.combine()
    out: dict = {"reanalyzed": sorted(redo), "model": {}, "pooled_x": {}, "reps": []}
    for test in new["model"]:
        out["model"][test] = [
            old["model"][test]["chi2"],
            new["model"][test]["chi2"],
            old["model"][test]["agrees"],
            new["model"][test]["agrees"],
        ]
    for key in ("lambda", "lambda_uncertainty", "predicted", "verdict"):
        out["pooled_x"][key] = [old["pooled_x"][key], new["pooled_x"][key]]
    for a, b in zip(old["reps"], new["reps"]):
        row = {
            "day": a["day"],
            "rep": a["rep"],
            "verdict": [a["verdict"], b["verdict"]],
            "tests": {},
        }
        for test in a["tests"]:
            ta, tb = a["tests"][test], b["tests"][test]
            row["tests"][test] = {
                k: [ta[k], tb[k]] for k in ("observed", "predicted", "bound", "agrees")
            }
        out["reps"].append(row)
    for name in sorted(redo):
        a = json.loads((RUNS / name / "analysis.json").read_text())
        b = json.loads((runs / name / "analysis.json").read_text())
        out.setdefault("jobs", {})[name] = {
            "f": [a["decoupled_dephasing"], b["decoupled_dephasing"]],
            "median_uncertainty": [
                a["idle_test"]["median_uncertainty"],
                b["idle_test"]["median_uncertainty"],
            ],
            "dephasing_from_memory": [a["dephasing_from_memory"], b["dephasing_from_memory"]],
            "agreement": {
                t: {
                    k: [a["agreement"]["tests"][t][k], b["agreement"]["tests"][t][k]]
                    for k in ("observed", "predicted", "from_f", "bound", "agrees")
                }
                for t in a["agreement"]["tests"]
            },
            "per_qubit_prediction": {
                basis: [
                    a["predicted_with_each_qubits_dephasing"]["fits"].get(basis),
                    b["predicted_with_each_qubits_dephasing"]["fits"].get(basis),
                ]
                for basis in "XZ"
            },
        }
    tag = "control" if control else "fixed"
    (RESULTS / f"idle_f_reanalysis{'_control' if control else ''}.json").write_text(
        json.dumps(out, indent=1, default=str)
    )
    shutil.copy2(runs / "combined.json", RESULTS / f"idle_f_combined_{tag}.json")
    print(json.dumps({k: out[k] for k in ("model", "pooled_x")}, indent=1))


def evaluate(runs: Path) -> dict:
    folders = sorted(p for p in runs.glob("ibm_fez-2026-10-*-r*") if (p / "analysis.json").exists())
    infos = {f.name: json.loads((f / "run.json").read_text()) for f in folders}
    per_job = {f.name: PQ.job_tests(f, None, None) for f in folders}
    out: dict = {"jobs": {}}
    for f in folders:
        a = json.loads((f / "analysis.json").read_text())
        dm = a["dephasing_from_memory"]
        out["jobs"][f.name] = {
            "f": a["decoupled_dephasing"],
            "sigma_f": a["idle_test"]["median_uncertainty"],
            "memory_f_agrees": dm.get("eps3_agrees", dm.get("eps5_agrees")),
        }
    for version in ("median_f", "per_qubit_f"):
        reps, model = PQ.combine({n: v[version] for n, v in per_job.items()}, infos)
        out[version] = {
            "chi2": {t: model[t]["chi2"] for t in TESTS},
            "reps_agreeing": {t: model[t]["reps_agreeing"] for t in TESTS},
            "all_three_X": sum(all(e["tests"][t]["agrees"] for t in TESTS[:3]) for e in reps),
            "pred_lambda_mean": {
                p: float(
                    np.mean([e["tests"]["X lambda"]["predicted"] for e in reps if e["rep"] == p])
                )
                for p in (1, 2)
            },
            "z": {
                f"{e['day']} r{e['rep']}": {t: e["tests"][t]["z"] for t in TESTS[:3]} for e in reps
            },
        }
    comb = json.loads((runs / "combined.json").read_text())
    out["pooled_x"] = {
        k: comb["pooled_x"][k] for k in ("lambda", "lambda_uncertainty", "predicted", "verdict")
    }
    out["verdicts"] = [e["verdict"] for e in comb["reps"]]
    return out


def compare(control_runs: Path, fixed_runs: Path) -> None:
    control, fixed = evaluate(control_runs), evaluate(fixed_runs)
    print("job                         f ctrl -> fix     sigma_f ctrl -> fix   memory-f agrees")
    for name in control["jobs"]:
        c, x = control["jobs"][name], fixed["jobs"][name]
        print(
            f"{name:27s} {c['f']:.4f} -> {x['f']:.4f}   {c['sigma_f']:.4f} -> {x['sigma_f']:.4f}   "
            f"{c['memory_f_agrees']} -> {x['memory_f_agrees']}"
        )
    for version in ("median_f", "per_qubit_f"):
        c, x = control[version], fixed[version]
        print(f"\n[{version}]")
        for t in TESTS:
            print(
                f"  {t:9s} chi2 {c['chi2'][t]:8.2f} -> {x['chi2'][t]:8.2f}   reps agreeing {c['reps_agreeing'][t]} -> {x['reps_agreeing'][t]}"
            )
    print("\npooled X:", control["pooled_x"], "->", fixed["pooled_x"])
    print("verdicts unchanged:", control["verdicts"] == fixed["verdicts"], set(fixed["verdicts"]))
    (RESULTS / "idle_f_compare.json").write_text(
        json.dumps({"control": control, "fixed": fixed}, indent=1, default=float)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="rerun the analysis of the named jobs in a scratch copy")
    r.add_argument("scratch", type=Path)
    r.add_argument("jobs", nargs="+")
    r.add_argument("--control", action="store_true", help="leave the variance as it ran")
    c = sub.add_parser("compare", help="set a control rerun against a corrected one")
    c.add_argument("control_runs", type=Path)
    c.add_argument("fixed_runs", type=Path)
    a = ap.parse_args()
    if a.command == "run":
        rerun(a.scratch, set(a.jobs), a.control)
    else:
        compare(a.control_runs, a.fixed_runs)


if __name__ == "__main__":
    main()
