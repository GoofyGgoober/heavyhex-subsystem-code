"""Corrected-XY-variance reanalysis against an unpatched control rerun in the same environment.

Both scratch runs directories come from reanalyze.py (corrected, and XY_CONTROL=1). Comparing
the corrected run with the control, not with the committed analysis.json, isolates the effect
of the variance fix from Monte Carlo differences of the prediction between environments.

    python compare.py <control scratch>/runs <corrected scratch>/runs
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "docs/diagnosis/revision/perqubit_f"))

import perqubit_f as PQ  # noqa: E402

TESTS = ("X lambda", "X eps3", "X eps5", "Z eps3", "Z eps5", "Z lambda")


def evaluate(runs: Path) -> dict:
    folders = sorted(p for p in runs.glob("ibm_fez-2026-10-*-r*") if (p / "analysis.json").exists())
    infos = {f.name: json.loads((f / "run.json").read_text()) for f in folders}
    per_job = {f.name: PQ.job_tests(f, None, None) for f in folders}
    out = {"jobs": {}}
    for f in folders:
        a = json.loads((f / "analysis.json").read_text())
        dm = a["dephasing_from_memory"]
        out["jobs"][f.name] = {"f": a["decoupled_dephasing"], "sigma_f": a["idle_test"]["median_uncertainty"],
                               "memory_f_agrees": dm.get("eps3_agrees", dm.get("eps5_agrees"))}
    for version in ("median_f", "per_qubit_f"):
        reps, model = PQ.combine({n: v[version] for n, v in per_job.items()}, infos)
        out[version] = {
            "chi2": {t: model[t]["chi2"] for t in TESTS},
            "reps_agreeing": {t: model[t]["reps_agreeing"] for t in TESTS},
            "all_three_X": sum(all(e["tests"][t]["agrees"] for t in TESTS[:3]) for e in reps),
            "pred_lambda_mean": {p: float(np.mean([e["tests"]["X lambda"]["predicted"] for e in reps if e["rep"] == p]))
                                 for p in (1, 2)},
            "z": {f"{e['day']} r{e['rep']}": {t: e["tests"][t]["z"] for t in TESTS[:3]} for e in reps},
        }
    comb = json.loads((runs / "combined.json").read_text())
    out["pooled_x"] = {k: comb["pooled_x"][k] for k in ("lambda", "lambda_uncertainty", "predicted", "verdict")}
    out["verdicts"] = [e["verdict"] for e in comb["reps"]]
    return out


def main() -> None:
    control, fixed = evaluate(Path(sys.argv[1])), evaluate(Path(sys.argv[2]))
    print("job                         f ctrl -> fix     sigma_f ctrl -> fix   memory-f agrees")
    for name in control["jobs"]:
        c, x = control["jobs"][name], fixed["jobs"][name]
        print(f"{name:27s} {c['f']:.4f} -> {x['f']:.4f}   {c['sigma_f']:.4f} -> {x['sigma_f']:.4f}   {c['memory_f_agrees']} -> {x['memory_f_agrees']}")
    print("memory-f agreements:", sum(bool(v["memory_f_agrees"]) for v in control["jobs"].values()),
          "->", sum(bool(v["memory_f_agrees"]) for v in fixed["jobs"].values()))
    for version in ("median_f", "per_qubit_f"):
        c, x = control[version], fixed[version]
        print(f"\n[{version}]")
        for t in TESTS:
            print(f"  {t:9s} chi2 {c['chi2'][t]:8.2f} -> {x['chi2'][t]:8.2f}   reps agreeing {c['reps_agreeing'][t]} -> {x['reps_agreeing'][t]}")
        print(f"  reps agreeing on all three X: {c['all_three_X']} -> {x['all_three_X']}")
        print(f"  mean predicted Lambda_X placement 1: {c['pred_lambda_mean'][1]:.4f} -> {x['pred_lambda_mean'][1]:.4f};"
              f" placement 2: {c['pred_lambda_mean'][2]:.4f} -> {x['pred_lambda_mean'][2]:.4f}")
    print("\npooled X:", control["pooled_x"], "->", fixed["pooled_x"])
    print("verdicts unchanged:", control["verdicts"] == fixed["verdicts"], set(fixed["verdicts"]))
    (HERE / "compare.json").write_text(json.dumps({"control": control, "fixed": fixed}, indent=1, default=float))


if __name__ == "__main__":
    main()
