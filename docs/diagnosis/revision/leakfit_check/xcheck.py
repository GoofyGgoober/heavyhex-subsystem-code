"""Cross-check of the two exact simulators on real circuits with leakage on.

leakage_exact/exact_sim.py (used for every simulation of these checks) against
tableau_fast.py (bit-identical to leaky_fix/tableau_ref.py), on the 8-round X- and
Z-memory circuits of 2026-10-08-r1 (both patches) at (leak 0.1%, partner X error 25%,
transfer 5%), 8000 shots each, independent random numbers: per-detector firing rates
(two-sample z, chi-square per detector) and the fit's statistics, pooled and per slot,
with shot-bootstrap standard errors (200 replicates) and the z of each difference.

    python docs/diagnosis/revision/leakfit_check/xcheck.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import exact_cache as E
import fitstats as S
import roundcond_fit as RC

HERE = Path(__file__).resolve().parent
JOB = "2026-10-08-r1"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    life, read = E.hmm_inputs()
    pt = E.Point(0.001, 0.25, 0.05, life, read)
    labels = {c: lab for c, (_, lab) in RC.device_cases().items()}
    out = {}
    for b in "XZ":
        x = E.load(JOB, b, 8, pt, sim="x")
        t = E.load(JOB, b, 8, pt, sim="t")
        px, pt_ = x.mean(0), t.mean(0)
        pool = (px + pt_) / 2
        se = np.sqrt(pool * (1 - pool) * (1 / len(x) + 1 / len(t)))
        z = (px - pt_) / np.where(se > 0, se, np.inf)
        sx, st = S.stats(x, labels[(JOB, b)]), S.stats(t, labels[(JOB, b)])
        keys = [k for k in sx if k != "profile"]
        rng = np.random.default_rng(0)  # shot-bootstrap standard errors of each statistic
        se = {}
        for name, arr in (("x", x), ("t", t)):
            pr = S.Prepared(arr, labels[(JOB, b)])
            reps = [pr.stats(np.bincount(rng.integers(0, len(arr), len(arr)), minlength=len(arr))) for _ in range(200)]
            se[name] = {k: float(np.std([r[k] for r in reps], ddof=1)) for k in keys}
        zstat = {k: (sx[k] - st[k]) / np.hypot(se["x"][k], se["t"][k]) for k in keys if np.hypot(se["x"][k], se["t"][k]) > 0}
        out[b] = {"detectors": len(z), "chi2_per_detector": float(np.mean(z ** 2)), "max_abs_z": float(np.abs(z).max()),
                  "mean_rate_exact_sim": float(px.mean()), "mean_rate_tableau": float(pt_.mean()),
                  "stats_exact_sim": {k: sx[k] for k in keys}, "stats_tableau": {k: st[k] for k in keys},
                  "stats_z": zstat}
        print(f"{JOB} {b}8: {len(z)} detectors, chi2/detector {np.mean(z**2):.2f} (1 expected), max |z| {np.abs(z).max():.1f}; "
              f"mean rate {px.mean():.4f} vs {pt_.mean():.4f}")
        print("   " + "  ".join(f"{k} {sx[k]:.4g}/{st[k]:.4g} (z {zstat.get(k, 0):+.1f})" for k in keys))
    (HERE / "xcheck.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
