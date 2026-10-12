"""Verifier's parametric bootstrap: for each of the 40 curves, draw B replicates from
the fitted binomial MLE curve and refit each with both estimators. Reports each
estimator's mean bias in eps and in Lambda (curves of a rep resampled independently),
and 68% percentile intervals. Seeds fixed. Usage: python v_boot.py [B]
"""
from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

from v_fit import ROUNDS, profile, wls

HERE = Path(__file__).resolve().parent


def fast_mle(k, N):
    res = minimize_scalar(lambda e: -profile(e, k, N)[0], bounds=(1e-4, 0.5 - 1e-6), method="bounded",
                          options=dict(xatol=1e-9, maxiter=500))
    return float(res.x)


def job(args):
    key, eps, A, N, B, seed = args
    rng = np.random.default_rng(seed)
    P = (1 - A * (1 - 2 * eps) ** ROUNDS) / 2
    K = rng.binomial(N, P, size=(B, len(ROUNDS)))
    w = np.array([wls(ROUNDS, k / N, N)["eps"] for k in K])
    m = np.array([fast_mle(k.astype(float), N) for k in K])
    nfail = int(np.sum(w >= 0.5))
    return key, w.tolist(), m.tolist(), nfail


def main():
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    res = json.loads((HERE / "v_results.json").read_text())
    tasks = []
    for i, r in enumerate(res):
        for j, (key, c) in enumerate(sorted(r["curves"].items())):
            tasks.append(((i, key), c["mle"]["eps"], c["mle"]["A"], c["N"], B, 1000 * i + j + 12345))
    out = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        for (i, key), w, m, nfail in ex.map(job, tasks):
            out[f"{i}:{key}"] = dict(wls=w, mle=m, nfail=nfail)
            print(i, key, "wls>=0.5:", nfail, flush=True)
    (HERE / "v_boot.json").write_text(json.dumps(out))


if __name__ == "__main__":
    main()
