"""Two estimators of the error per round eps from P(n) = (1 - A (1 - 2 eps)^n) / 2.

wls      : the prespecified fit (heavyhex.simulation.noisy.fit_per_round), re-implemented:
           weighted least squares of ln(1 - 2 P) = a + b n, weights from the binomial
           variance at the observed P, points with P <= 0 or P >= 0.5 dropped, the
           standard error of eps scaled by sqrt(chi2_red) when chi2_red > 1.
binomial : maximum likelihood with k_n ~ Binomial(N, P(n)) at every round count, no
           points dropped. A is profiled out exactly for each eps (the log-likelihood is
           concave in A at fixed eps), so the profile likelihood of eps is exact.

Everything is vectorised over leading axes so the parametric bootstrap can refit
thousands of replicates at once.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, sqrt

import numpy as np
from scipy.optimize import brentq
from scipy.stats import chi2

EPS_MAX = 0.5 - 1e-9  # q = 1 - 2 eps must stay positive
TINY = 1e-300


# ----------------------------------------------------------------------------- WLS
@dataclass
class WLS:
    eps: float
    se: float  # widened by sqrt(chi2_red) when > 1 (as prespecified)
    se_raw: float  # shot noise alone
    chi2_red: float
    A: float
    used: int  # points kept


def wls(rounds, p, shots) -> WLS:
    """The prespecified fit, re-implemented from its docstring and code."""
    p, n = np.asarray(p, float), np.asarray(rounds, float)
    keep = (p > 0) & (p < 0.5)
    if keep.sum() < 2:
        return WLS(0.5, float("nan"), float("nan"), float("nan"), float("nan"), int(keep.sum()))
    p, n = p[keep], n[keep]
    sigma = 2 * np.sqrt(p * (1 - p) / shots) / (1 - 2 * p)
    y = np.log(1 - 2 * p)
    w = 1 / sigma**2
    S, Sx, Sy, Sxx, Sxy = (
        w.sum(),
        (w * n).sum(),
        (w * y).sum(),
        (w * n * n).sum(),
        (w * n * y).sum(),
    )
    det = S * Sxx - Sx**2
    b = (S * Sxy - Sx * Sy) / det
    a = (Sxx * Sy - Sx * Sxy) / det
    var_b = S / det
    chi2_red = float(np.sum(((y - a - b * n) / sigma) ** 2) / (len(n) - 2)) if len(n) > 2 else 1.0
    se_raw = exp(b) / 2 * sqrt(var_b)
    return WLS(
        (1 - exp(b)) / 2, se_raw * sqrt(max(chi2_red, 1.0)), se_raw, chi2_red, exp(a), len(n)
    )


def wls_batch(rounds, p, shots):
    """wls over the rows of p (B x m). Returns eps (0.5 where < 2 points) and widened se."""
    p = np.asarray(p, float)
    n = np.broadcast_to(np.asarray(rounds, float), p.shape)
    keep = (p > 0) & (p < 0.5)
    pc = np.where(keep, p, 0.25)
    sigma = 2 * np.sqrt(pc * (1 - pc) / shots) / (1 - 2 * pc)
    y = np.log(1 - 2 * pc)
    w = np.where(keep, 1 / sigma**2, 0.0)
    S, Sx, Sy = w.sum(1), (w * n).sum(1), (w * y).sum(1)
    Sxx, Sxy = (w * n * n).sum(1), (w * n * y).sum(1)
    m = keep.sum(1)
    ok = m >= 2
    det = np.where(ok, S * Sxx - Sx**2, 1.0)
    b = np.where(ok, (S * Sxy - Sx * Sy) / det, np.log(TINY))
    a = np.where(ok, (Sxx * Sy - Sx * Sxy) / det, 0.0)
    var_b = np.where(ok, S / det, np.nan)
    resid = np.where(keep, ((y - a[:, None] - b[:, None] * n) / sigma) ** 2, 0.0).sum(1)
    chi2_red = np.where(m > 2, resid / np.maximum(m - 2, 1), 1.0)
    eps = np.where(ok, (1 - np.exp(b)) / 2, 0.5)
    se = np.exp(b) / 2 * np.sqrt(var_b) * np.sqrt(np.maximum(chi2_red, 1.0))
    return eps, np.where(ok, se, np.nan)


# ------------------------------------------------------------------------ binomial
def _loglik(P, k, N):
    P = np.clip(P, TINY, 1 - 1e-16)
    return k * np.log(P) + (N - k) * np.log1p(-P)


def best_A(q, k, N, n, iters: int = 64):
    """argmax_A of the log-likelihood at fixed q = 1 - 2 eps, by bisection on the score.

    q has any shape S; k has shape S + (m,) (or broadcastable). A lies in [0, 1/q): A > 0
    keeps P < 1/2 at every n, A < 1/q keeps P(1) > 0. The score is decreasing in A
    (the log-likelihood is concave in A), so bisection is exact.
    """
    q = np.asarray(q, float)
    qn = q[..., None] ** n  # S + (m,)
    lo = np.zeros(q.shape)
    hi = (1 - 1e-12) / q

    def score(A):
        P = (1 - A[..., None] * qn) / 2
        P = np.clip(P, 1e-15, 1 - 1e-15)
        return (-qn / 2 * (k / P - (N - k) / (1 - P))).sum(-1)

    s0 = score(lo)
    for _ in range(iters):
        mid = (lo + hi) / 2
        pos = score(mid) > 0
        lo = np.where(pos, mid, lo)
        hi = np.where(pos, hi, mid)
    A = (lo + hi) / 2
    return np.where(s0 <= 0, 0.0, A)  # data at or above 1/2 on balance: A = 0


def profile(eps, k, N, n):
    """Profile log-likelihood of eps (A maximised), and the maximising A."""
    q = 1 - 2 * np.asarray(eps, float)
    A = best_A(q, k, N, n)
    P = (1 - A[..., None] * q[..., None] ** n) / 2
    return _loglik(P, k, N).sum(-1), A


GRID = np.concatenate([np.linspace(1e-4, 0.45, 900), np.linspace(0.45, EPS_MAX, 200)[1:]])


def mle_batch(k, N, n, grid=GRID, refine: int = 40):
    """MLE of (eps, A) for each row of k (B x m): grid search then golden section."""
    k = np.asarray(k, float)
    B = k.shape[0]
    ll = profile(np.broadcast_to(grid, (B, len(grid))), k[:, None, :], N, n)[0]
    i = ll.argmax(1)
    lo = grid[np.maximum(i - 1, 0)]
    hi = grid[np.minimum(i + 1, len(grid) - 1)]
    g = (sqrt(5) - 1) / 2
    c, d = hi - g * (hi - lo), lo + g * (hi - lo)
    fc, fd = profile(c, k, N, n)[0], profile(d, k, N, n)[0]
    for _ in range(refine):
        left = fc > fd
        hi = np.where(left, d, hi)
        lo = np.where(left, lo, c)
        c_new, d_new = hi - g * (hi - lo), lo + g * (hi - lo)
        # recompute both (cheap, avoids index bookkeeping)
        c, d = c_new, d_new
        fc, fd = profile(c, k, N, n)[0], profile(d, k, N, n)[0]
    eps = (lo + hi) / 2
    llmax, A = profile(eps, k, N, n)
    return eps, A, llmax


@dataclass
class Binomial:
    eps: float
    A: float
    loglik: float
    deviance: float  # against the saturated model, m - 2 degrees of freedom
    p_gof: float
    pearson_red: float  # Pearson X^2 / (m - 2): the overdispersion factor phi
    lo: float  # 68.27% profile-likelihood interval (Delta(-2 ln L) = 1)
    hi: float
    lo_q: float  # quasi-likelihood interval: Delta(-2 ln L) = max(phi, 1)
    hi_q: float
    at_boundary: bool  # upper end hit eps = 1/2


def _crossing(f, a, b):
    """Root of f in [a, b] if it changes sign, else None."""
    fa, fb = f(a), f(b)
    if np.sign(fa) == np.sign(fb):
        return None
    return brentq(f, a, b, xtol=1e-10)


def binomial(rounds, k, N) -> Binomial:
    n = np.asarray(rounds, float)
    k = np.asarray(k, float)
    eps, A, ll = (float(x[0]) for x in mle_batch(k[None, :], N, n))
    P = (1 - A * (1 - 2 * eps) ** n) / 2
    sat = _loglik(np.clip(k / N, 1e-15, 1 - 1e-15), k, N).sum()
    dev = float(2 * (sat - ll))
    dof = len(n) - 2
    pearson = float(((k - N * P) ** 2 / (N * P * (1 - P))).sum() / dof)
    phi = max(pearson, 1.0)

    def prof(e):
        return float(profile(np.array(e), k, N, n)[0])

    def interval(level):
        f = lambda e: 2 * (ll - prof(e)) - level  # noqa: E731
        lo = _crossing(f, 1e-6, eps) if eps > 1e-6 else 0.0
        hi = _crossing(f, eps, EPS_MAX)
        return (0.0 if lo is None else lo), (0.5 if hi is None else hi), hi is None

    lo, hi, edge = interval(1.0)
    lo_q, hi_q, edge_q = interval(phi)
    return Binomial(
        eps, A, ll, dev, float(chi2.sf(dev, dof)), pearson, lo, hi, lo_q, hi_q, edge or edge_q
    )


def lambda_profile(rounds, k3, N3, k5, N5, phi3=1.0, phi5=1.0, level=1.0):
    """MLE of Lambda = eps3 / eps5 and its profile interval from the joint likelihood of
    the two independent curves, each divided by its overdispersion factor phi
    (quasi-likelihood; phi = 1 is the pure binomial likelihood).

    l(Lambda) = max over eps5 of l3(Lambda eps5) / phi3 + l5(eps5) / phi5.
    """
    n = np.asarray(rounds, float)
    k3, k5 = np.asarray(k3, float), np.asarray(k5, float)
    e5grid = np.concatenate([np.linspace(1e-3, 0.45, 600), np.linspace(0.45, EPS_MAX, 100)[1:]])

    def joint(lam):
        e5 = e5grid[e5grid * lam < EPS_MAX]
        l3 = profile(lam * e5, k3, N3, n)[0] / phi3
        l5 = profile(e5, k5, N5, n)[0] / phi5
        tot = l3 + l5
        j = int(tot.argmax())
        # golden refine around the grid maximum
        a, b = e5[max(j - 1, 0)], e5[min(j + 1, len(e5) - 1)]
        g = (sqrt(5) - 1) / 2

        def f(e):
            return float(
                profile(np.array(lam * e), k3, N3, n)[0] / phi3
                + profile(np.array(e), k5, N5, n)[0] / phi5
            )

        c, d = b - g * (b - a), a + g * (b - a)
        fc, fd = f(c), f(d)
        for _ in range(50):
            if fc > fd:
                b, d, fd = d, c, fc
                c = b - g * (b - a)
                fc = f(c)
            else:
                a, c, fc = c, d, fd
                d = a + g * (b - a)
                fd = f(d)
        return max(fc, fd, float(tot[j]))

    # the joint maximum is at the separate maxima
    e3 = binomial_point(n, k3, N3)
    e5 = binomial_point(n, k5, N5)
    lam_hat = e3 / e5
    lmax = joint(lam_hat)
    f = lambda lam: 2 * (lmax - joint(lam)) - level  # noqa: E731
    lo = _crossing(f, lam_hat * 0.2, lam_hat)
    hi = _crossing(f, lam_hat, lam_hat * 5)
    return lam_hat, (np.nan if lo is None else lo), (np.nan if hi is None else hi)


def binomial_point(n, k, N) -> float:
    return float(mle_batch(np.asarray(k, float)[None, :], N, np.asarray(n, float))[0][0])
