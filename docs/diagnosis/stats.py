"""Robustness checks on the registered result, all exploratory:

1. Λ per placement: the reps sit on two overlapping placements, the same every day.
2. Pooling on ln Λ, and unweighted, beside the registered inverse-variance pooling of Λ.
3. Day-level pooling: each day's two reps first, then the days (Student's t).
4. The agreement test with the calibration's day-to-day scatter in the predicted variance:
   the spread of one placement's predictions over the days, which the registered test
   leaves out (it allows only shot noise, f and the before/after gap within a job).
5. f: the registered median over the job's data qubits beside the inverse-variance mean.

    python docs/diagnosis/stats.py
"""

from __future__ import annotations

import json
from collections import defaultdict
from math import sqrt

import numpy as np
from scipy.stats import chi2, t as student_t

from common import REPO, jobs


def pooled(values, sigmas):
    w = 1 / np.asarray(sigmas) ** 2
    m = float((w * values).sum() / w.sum()); se = float(sqrt(1 / w.sum()))
    k = len(values); spread = float(((np.asarray(values) - m) ** 2 * w).sum() / (k - 1)) if k > 1 else 1.0
    return m, se * sqrt(max(spread, 1.0)), spread


def main() -> None:
    c = json.loads((REPO / "runs/combined.json").read_text())
    reps = c["reps"]
    lam = np.array([r["tests"]["X lambda"]["observed"] for r in reps])
    sig = np.array([r["tests"]["X lambda"]["observed_uncertainty"] for r in reps])
    pred = np.array([r["tests"]["X lambda"]["predicted"] for r in reps])
    out: dict = {"registered": c["pooled_x"]}
    print(f"registered: Λ {c['pooled_x']['lambda']:.3f} ± {c['pooled_x']['lambda_uncertainty']:.3f}, predicted {c['pooled_x']['predicted']:.3f} ({len(reps)} reps)")

    # 1. per placement
    groups = defaultdict(list)
    for i, r in enumerate(reps):
        groups[f"rep {r['rep']}: d=5 at {tuple(r['spots']['5'][0])}"].append(i)
    out["per_placement"] = {}
    for g, ix in groups.items():
        m, se, s = pooled(lam[ix], sig[ix]); mp = float(np.mean(pred[ix]))
        out["per_placement"][g] = {"lambda": m, "se": se, "predicted_mean": mp, "days": len(ix)}
        print(f"  {g}: Λ {m:.3f} ± {se:.3f} over {len(ix)} days (predictions averaged {mp:.3f}); χ²/dof between days {s:.2f}")

    # 2. pooling choices
    lnl, lns = np.log(lam), sig / lam
    m, se, _ = pooled(lnl, lns)
    out["ln_pooled"] = {"lambda": float(np.exp(m)), "low": float(np.exp(m - se)), "high": float(np.exp(m + se))}
    out["unweighted"] = {"lambda": float(lam.mean()), "se": float(lam.std(ddof=1) / sqrt(len(lam)))}
    print(f"  pooled on ln Λ: {np.exp(m):.3f} (-{np.exp(m)-np.exp(m-se):.3f}/+{np.exp(m+se)-np.exp(m):.3f}); unweighted mean {lam.mean():.3f} ± {out['unweighted']['se']:.3f}")

    # 3. day level
    days = defaultdict(list)
    for i, r in enumerate(reps):
        days[r["day"]].append(i)
    dm = [pooled(lam[ix], sig[ix])[0] for ix in days.values()]
    dmean, dse = float(np.mean(dm)), float(np.std(dm, ddof=1) / sqrt(len(dm)))
    tcrit = float(student_t.ppf(0.95, len(dm) - 1))
    out["day_level"] = {"by_day": dict(zip(days, dm)), "lambda": dmean, "se": dse, "t": tcrit,
                        "d5_worse": bool(dmean + tcrit * dse < 1)}
    print(f"  day level: {', '.join(f'{v:.3f}' for v in dm)} -> {dmean:.3f} ± {dse:.3f}; one-sided t({len(dm)-1}) {tcrit:.2f}: "
          f"{'d = 5 worse' if dmean + tcrit * dse < 1 else 'undecided'}")

    # 4. agreement test with calibration scatter
    print("\n  agreement over the reps, χ² against the 95% bar, as registered -> with the calibration's day-to-day scatter:")
    rep1 = [i for i, r in enumerate(reps) if r["rep"] == 1]  # one placement, five calibrations
    out["agreement_with_scatter"] = {}
    bar = float(chi2.ppf(0.95, len(reps)))
    for q in ("X lambda", "X eps3", "X eps5", "Z lambda", "Z eps3", "Z eps5"):
        o = np.array([r["tests"][q]["observed"] for r in reps]); p = np.array([r["tests"][q]["predicted"] for r in reps])
        half = np.array([r["tests"][q]["bound"] / 2 for r in reps])
        scatter = float(np.std(p[rep1], ddof=1) / np.mean(p[rep1]))  # relative day-to-day spread of one placement's prediction
        sd_cal = scatter * p
        before = float((((o - p) / half) ** 2).sum()); after = float((((o - p) ** 2) / (half ** 2 + sd_cal ** 2)).sum())
        out["agreement_with_scatter"][q] = {"chi2": before, "chi2_with_scatter": after, "bar": bar, "relative_scatter": scatter}
        print(f"    {q:9s} {before:7.1f} -> {after:6.1f}  (bar {bar:.1f}; prediction scatter {100*scatter:.0f}% of its value) "
              f"{'passes' if after < bar else 'fails'}")

    # 5. f
    print("\n  f per job: registered median | inverse-variance mean | χ²/dof across qubits")
    out["f"] = {}
    for j in jobs():
        it = j.analysis["idle_test"]
        out["f"][j.name] = {"median": it["median"], "weighted_mean": it["uniformity"]["weighted_mean"],
                           "chi2_per_dof": it["uniformity"]["chi2_per_dof"], "quartiles": it["uniformity"]["quartiles"]}
        print(f"    {j.name:15s} {it['median']:.2f} ± {it['median_uncertainty']:.2f} | {it['uniformity']['weighted_mean']:.2f} | {it['uniformity']['chi2_per_dof']:.0f}")
    (REPO / "docs/diagnosis/results/stats.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
