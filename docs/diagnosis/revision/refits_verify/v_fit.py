"""Verifier's own fitting library (written independently of ../refits/fitlib.py).

Model: P(n) = (1 - A r^n) / 2, r = 1 - 2 eps.
  * wls(): the prespecified weighted fit of ln(1 - 2P) on n, written out with explicit
    normal equations (no polyfit), dropping P <= 0 and P >= 0.5, error scaled by
    sqrt(max(1, chi2_red)).
  * Binomial MLE with A profiled out exactly (log-likelihood concave in A), profile
    likelihood intervals for eps, and the joint d=3/d=5 profile for Lambda.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq, minimize_scalar

ROUNDS = np.array([1, 2, 3, 4, 6, 8], dtype=float)
TINY = 1e-12


# ---------------------------------------------------------------- weighted log fit
def wls(n, p, N):
    n = np.asarray(n, float)
    p = np.asarray(p, float)
    keep = (p > 0) & (p < 0.5)
    if keep.sum() < 2:
        return dict(eps=0.5, se=float("nan"), chi2_red=float("nan"), A=float("nan"), used=int(keep.sum()))
    n, p = n[keep], p[keep]
    y = np.log(1 - 2 * p)
    var = 4 * p * (1 - p) / (N * (1 - 2 * p) ** 2)
    w = 1 / var
    S, Sx, Sxx = w.sum(), (w * n).sum(), (w * n * n).sum()
    Sy, Sxy = (w * y).sum(), (w * n * y).sum()
    D = S * Sxx - Sx * Sx
    b = (S * Sxy - Sx * Sy) / D
    a = (Sxx * Sy - Sx * Sxy) / D
    var_b = S / D
    m = len(n)
    chi2_red = float((w * (y - a - b * n) ** 2).sum() / (m - 2)) if m > 2 else 1.0
    eps = (1 - math.exp(b)) / 2
    se = math.exp(b) / 2 * math.sqrt(var_b) * math.sqrt(max(chi2_red, 1.0))
    return dict(eps=eps, se=se, chi2_red=chi2_red, A=math.exp(a), used=int(m))


# ---------------------------------------------------------------- binomial likelihood
def loglik(eps, A, k, N, n=ROUNDS):
    r = 1 - 2 * eps
    P = (1 - A * r ** n) / 2
    if np.any(P <= 0) or np.any(P >= 1):
        return -np.inf
    return float(np.sum(k * np.log(P) + (N - k) * np.log1p(-P)))


def best_A(eps, k, N, n=ROUNDS):
    """A maximising the likelihood at fixed eps (concave in A); valid |A r^n| < 1."""
    r = 1 - 2 * eps
    hi = 1 / r ** n.min() * (1 - 1e-12)
    lo = -hi
    k = np.asarray(k, float)
    rn = r ** n

    def dll(A):
        P = (1 - A * rn) / 2
        # d/dA [k ln P + (N-k) ln(1-P)] = (-rn/2) [k/P - (N-k)/(1-P)]
        return float(np.sum(-rn / 2 * (k / P - (N - k) / (1 - P))))

    # derivative is decreasing in A (concavity); bracket root
    dlo, dhi = dll(lo), dll(hi)
    if dlo <= 0:
        return lo
    if dhi >= 0:
        return hi
    return brentq(dll, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)


def profile(eps, k, N, n=ROUNDS):
    A = best_A(eps, k, N, n)
    return loglik(eps, A, k, N, n), A


def mle(k, N, n=ROUNDS):
    k = np.asarray(k, float)
    grid = np.linspace(0.002, 0.498, 249)
    vals = np.array([profile(e, k, N, n)[0] for e in grid])
    i = int(np.argmax(vals))
    lo, hi = grid[max(i - 1, 0)] if i > 0 else 1e-6, grid[min(i + 1, len(grid) - 1)] if i < len(grid) - 1 else 0.5 - 1e-6
    res = minimize_scalar(lambda e: -profile(e, k, N, n)[0], bounds=(lo, hi), method="bounded",
                          options=dict(xatol=1e-13, maxiter=500))
    eps = float(res.x)
    ll, A = profile(eps, k, N, n)
    P = (1 - A * (1 - 2 * eps) ** n) / 2
    pearson = float(np.sum((k - N * P) ** 2 / (N * P * (1 - P))))
    with np.errstate(divide="ignore", invalid="ignore"):
        t1 = np.where(k > 0, k * np.log(k / (N * P)), 0.0)
        t2 = np.where(N - k > 0, (N - k) * np.log((N - k) / (N * (1 - P))), 0.0)
    deviance = float(2 * np.sum(t1 + t2))
    dof = len(n) - 2
    phi = max(1.0, pearson / dof)
    return dict(eps=eps, A=float(A), ll=ll, pearson=pearson, deviance=deviance, dof=dof, phi=phi)


def eps_interval(k, N, fit, level=1.0, n=ROUNDS):
    """{eps: 2 (ll_max - prof(eps)) <= level}."""
    k = np.asarray(k, float)
    e0, l0 = fit["eps"], fit["ll"]

    def g(e):
        return 2 * (l0 - profile(e, k, N, n)[0]) - level

    lo_edge, hi_edge = 1e-7, 0.5 - 1e-7
    lo = brentq(g, lo_edge, e0, xtol=1e-12) if g(lo_edge) > 0 else lo_edge
    hi = brentq(g, e0, hi_edge, xtol=1e-12) if g(hi_edge) > 0 else hi_edge
    return lo, hi


# ---------------------------------------------------------------- Lambda profile
def lambda_profile(k3, k5, N3, N5, phi3=1.0, phi5=1.0, n=ROUNDS):
    k3, k5 = np.asarray(k3, float), np.asarray(k5, float)
    f3, f5 = mle(k3, N3, n), mle(k5, N5, n)
    lam_hat = f3["eps"] / f5["eps"]
    qmax = f3["ll"] / phi3 + f5["ll"] / phi5

    def qprof(lam):
        top = min(0.5 - 1e-9, (0.5 - 1e-9) / lam)

        def neg(e5):
            return -(profile(lam * e5, k3, N3, n)[0] / phi3 + profile(e5, k5, N5, n)[0] / phi5)

        # coarse then refine
        grid = np.linspace(1e-4, top, 120)
        v = np.array([neg(e) for e in grid])
        i = int(np.argmin(v))
        a = grid[max(i - 1, 0)]
        b = grid[min(i + 1, len(grid) - 1)]
        res = minimize_scalar(neg, bounds=(a, b), method="bounded", options=dict(xatol=1e-12))
        return -float(res.fun)

    def g(lam):
        return 2 * (qmax - qprof(lam)) - 1.0

    lo = brentq(g, lam_hat * 0.3, lam_hat, xtol=1e-10) if g(lam_hat * 0.3) > 0 else float("nan")
    hi = brentq(g, lam_hat, lam_hat * 3, xtol=1e-10) if g(lam_hat * 3) > 0 else float("nan")
    return lam_hat, lo, hi
