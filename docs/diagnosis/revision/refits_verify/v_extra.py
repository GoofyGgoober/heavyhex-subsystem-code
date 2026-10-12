"""Verifier extras: (1) per-rep one-sided 95% upper limits on Lambda_X from the joint
quasi-profile (2 dlnL = 1.645^2), i.e. the decision rule without the symmetric-sigma
approximation; (2) sensitivity of the MLE eps to constraining A <= 1 (P(0) >= 0)."""
import json
from pathlib import Path

import numpy as np
from scipy.optimize import brentq, minimize_scalar

import v_fit
from v_fit import loglik, mle, profile

HERE = Path(__file__).resolve().parent
res = json.loads((HERE / "v_results.json").read_text())


def lam_upper(k3, k5, N3, N5, phi3, phi5, level):
    f3, f5 = mle(k3, N3), mle(k5, N5)
    lam_hat = f3["eps"] / f5["eps"]
    qmax = f3["ll"] / phi3 + f5["ll"] / phi5

    def qprof(lam):
        top = min(0.5 - 1e-9, (0.5 - 1e-9) / lam)
        neg = lambda e5: -(profile(lam * e5, k3, N3)[0] / phi3 + profile(e5, k5, N5)[0] / phi5)
        grid = np.linspace(1e-4, top, 120)
        v = np.array([neg(e) for e in grid])
        i = int(np.argmin(v))
        r = minimize_scalar(neg, bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 119)]), method="bounded",
                            options=dict(xatol=1e-12))
        return -float(r.fun)

    g = lambda lam: 2 * (qmax - qprof(lam)) - level
    return lam_hat, brentq(g, lam_hat, lam_hat * 3, xtol=1e-8)


print("(1) Lambda_X one-sided 95% upper limit (quasi profile, 2dlnL = 2.706):")
ups = []
for r in res:
    c3, c5 = r["curves"]["X3"], r["curves"]["X5"]
    lam, up = lam_upper(np.array(c3["k"], float), np.array(c5["k"], float), c3["N"], c5["N"],
                        c3["mle"]["phi"], c5["mle"]["phi"], 1.645**2)
    ups.append(up)
    print(f"  {r['day']} r{r['rep']}: {lam:.3f}, upper {up:.3f}  (symmetric: {lam + 1.645 * r['lam']['X']['sigma_q']:.3f})")
print("  all upper limits < 1:", all(u < 1 for u in ups))


def mle_Ale1(k, N):
    def prof(e):
        A = min(v_fit.best_A(e, k, N), 1.0)
        return loglik(e, A, k, N)
    grid = np.linspace(0.002, 0.498, 249)
    v = np.array([prof(e) for e in grid])
    i = int(np.argmax(v))
    r = minimize_scalar(lambda e: -prof(e), bounds=(grid[max(i - 1, 0)], grid[min(i + 1, 248)]), method="bounded",
                        options=dict(xatol=1e-12))
    return float(r.x)


print("\n(2) MLE with A <= 1 (only the 12 Z curves have A > 1):")
for key in ("Z3", "Z5"):
    free = [r["curves"][key]["mle"]["eps"] for r in res]
    con = [mle_Ale1(np.array(r["curves"][key]["k"], float), r["curves"][key]["N"]) for r in res]
    print(f"  {key}: mean free {100*np.mean(free):.2f}%  A<=1 {100*np.mean(con):.2f}%")
lz_free = np.mean([r["lam"]["Z"]["mle"] for r in res])
lz_con = np.mean([mle_Ale1(np.array(r["curves"]["Z3"]["k"], float), 2500) / mle_Ale1(np.array(r["curves"]["Z5"]["k"], float), 2500) for r in res])
print(f"  mean Lambda_Z: free {lz_free:.3f}  A<=1 {lz_con:.3f}")
