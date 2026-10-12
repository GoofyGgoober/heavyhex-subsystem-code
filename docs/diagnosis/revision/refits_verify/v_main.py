"""Verifier: refit all 40 curves with both estimators, pool, and compare with the
stored repo numbers. Writes v_results.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from v_fit import ROUNDS, eps_interval, lambda_profile, mle, wls

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
reps = json.loads((HERE / "v_counts.json").read_text())


def fit_curve(k, N, n=ROUNDS):
    k = np.asarray(k, float)
    w = wls(n, k / N, N)
    m = mle(k, N, n)
    plo, phi_ = eps_interval(k, N, m, 1.0, n)
    qlo, qhi = eps_interval(k, N, m, m["phi"], n)
    m.update(prof=(plo, phi_), quasi=(qlo, qhi), sigma_q=(qhi - qlo) / 2, sigma_p=(phi_ - plo) / 2)
    return w, m


results = []
for r in reps:
    row = {"day": r["day"], "day_index": r["day_index"], "rep": r["rep"], "curves": {}, "lam": {}}
    for key, c in r["curves"].items():
        w, m = fit_curve(c["k"], c["N"])
        # stored check
        st = c["stored_fit"]
        row["curves"][key] = dict(wls=w, mle=m, stored=st, k=c["k"], N=c["N"])
    for b in "XZ":
        c3, c5 = row["curves"][f"{b}3"], row["curves"][f"{b}5"]
        lw = c3["wls"]["eps"] / c5["wls"]["eps"]
        sw = lw * math.hypot(c3["wls"]["se"] / c3["wls"]["eps"], c5["wls"]["se"] / c5["wls"]["eps"])
        lam_hat, qlo, qhi = lambda_profile(c3["k"], c5["k"], c3["N"], c5["N"], c3["mle"]["phi"], c5["mle"]["phi"])
        _, plo, phi_ = lambda_profile(c3["k"], c5["k"], c3["N"], c5["N"], 1.0, 1.0)
        row["lam"][b] = dict(wls=lw, wls_se=sw, mle=lam_hat, quasi=(qlo, qhi), prof=(plo, phi_),
                             sigma_q=(qhi - qlo) / 2)
    results.append(row)
    print(r["day"], r["rep"], flush=True)

(HERE / "v_results.json").write_text(json.dumps(results, indent=1))
print("done")
