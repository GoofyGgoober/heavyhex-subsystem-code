"""The X-memory agreement tests when the ten repetitions are not counted as ten independent ones;
every raw count against the model; and what the data say about Lambda_X within a practical
margin (Sec. agreement, App. stats).

A repetition is one day and placement. The ten share more than independent replications would:
the three jobs of a day ran within four minutes on one calibration snapshot, the two distance-5
patches share 51 of their 65 qubits and were the same every day, and placement 2's distance-3
patch lies inside both of them. Under the prespecified null (the prediction right within the
stated errors) the ten z are still independent, since the shots, the simulations and each job's
f are; the dependence matters once the prediction may miss by a day or by a patch.

  structure    the qubits each pair of patches shares; the calibration of each day's jobs
  blocks       X eps3, eps5 and Lambda, both fits:
                 prespecified  sum of the ten z^2 against chi2 with 10 dof (as stored)
                 correlated    the same statistic if the two placements' z on a day had
                               correlation rho, weights 1 +- rho with 5 dof each (Imhof): p over
                               rho; at rho = 1 it is T/2 against chi2 with 5 dof
                 per placement each placement's five days (5 dof): five jobs on five days,
                               independent under the null whatever the other placement shares;
                               offset (1 dof) and scatter (4 dof)
                 day blocks    the common offset when each day may miss by its own amount: the
                               two placements averaged within a day, a t test over the five days
                               (4 dof) on z and on ln(O/M), and the exact sign-flip test over the
                               32 sign patterns of the five day means
                 hierarchical  ln(O/M) = mu + a_day + e with a_day ~ N(0, tau_day^2) and
                               e ~ N(0, v + tau^2), v the stated variances: REML, and a
                               parametric-bootstrap likelihood-ratio test of mu = 0 whose null
                               keeps both variance components; also one mu per placement
                 effect sizes  O/M with 95% intervals, apart from any test: per placement (t over
                               days), both placements by day blocks, the hierarchical mu, and the
                               day-block bootstrap over all 5^5 resamples (too narrow with five
                               blocks; a check)
  counts       every curve's six raw counts k(n) of 2,500 against the model's P(n) at the job's
               f (200,000 simulated shots): per point z from the chip's and the simulator's
               binomial noise and the f term, sigma_f dP/df with dP/df the line through the
               f = 0 and f = 1 predictions (run._slopes's parabola is noisy near f = 0.95,
               sensitivity.py), shared by a job's points; chi2 per curve, round 1 apart from
               rounds 2-8, and without the points the model puts at P >= 0.45, where a change in
               eps hardly moves P (snr_10pct: the shift of P from 10% more eps, in chip standard
               errors); sums per basis, placement and distance; the excess k/N - P per round
               count with a t interval over the days; and a fit anchored to the model's own curve
               on rounds 2-8, P(n) = [1 - a r^(n-2) (1 - 2 P_model(n))] / 2 (no exponential form
               assumed): r is the device's fidelity per round relative to the model's, as an
               error per round eps = (1 - r q) / 2 against the model's (1 - q) / 2, q the
               model's own fidelity per round over rounds 2-8; a0 = a / r^2 the relative
               fidelity extrapolated to n = 0 (preparation and final readout); and round 1
               against this trend, its excess in pp and a likelihood ratio (1 dof) of round 1
               left free against round 1 on the trend: an early transient would show there
  equivalence  Lambda_X, O - M against margins of 0.05 and 0.10: two one-sided t tests per
               placement and by day blocks (90% intervals), the stated-error and prespecified
               pooled versions beside them, the smallest margin the data certify, and the
               chance of certifying 0.05 were the model exactly right

Reads runs/ (combined.json, run.json, job.json, calibration.json) and results/ (fits.json,
fits_counts.json, power.json, sensitivity.json); no simulation; seeded; 30 s in one process,
0.25 GB.

    python docs/analysis/dependence.py      # writes results/dependence.json
"""

from __future__ import annotations

import itertools
from datetime import datetime
from math import sqrt

import numpy as np
from scipy import integrate, stats
from scipy.optimize import brentq, minimize, minimize_scalar

import effect_sizes as ES
from common import day_number, job_folders, read, result, write_json

X_TESTS = ES.X_TESTS
FITS = ("binomial", "wls")  # binomial primary for effect sizes, WL the prespecified test
PLACEMENTS = (1, 2)
PRED, F0, F1 = "predicted_at_measured_dephasing", "frozen_f0.0", "frozen_f1.0"
NEAR_HALF = 0.45
MARGINS = (0.05, 0.10)
B = 2000  # parametric bootstrap replicates
MC = 200_000  # Monte Carlo draws for the equivalence power
SEED = 1
TAU = np.concatenate([np.linspace(0, 0.2, 41), np.linspace(0.2, 0.6, 17)[1:]])  # grid of tau


def floats(x) -> list:
    return [float(v) for v in x]


def day_means(x, day) -> np.ndarray:
    return np.array([np.mean(x[day == d]) for d in np.unique(day)])


# ------------------------------------------------------------------------------ structure
def structure() -> dict:
    folders = {f.name: f for f in job_folders()}
    runs = {n: read(f / "run.json") for n, f in folders.items()}
    chips = {n: {p: set(q) for p, q in r["chips"].items()} for n, r in runs.items()}
    days = sorted({r["rep"]["day"] for r in runs.values()})
    shared, same, minutes = {}, {}, {}
    for day in days:
        p1, a3, a5 = (chips[f"ibm_fez-{day}-{s}"] for s in ("r1", "r2-d3", "r2-d5"))
        pairs = {
            "d5 P1 & d5 P2": (p1["5"], a5["5"]),
            "d3 P2 & d5 P1": (a3["3"], p1["5"]),
            "d3 P2 & d5 P2": (a3["3"], a5["5"]),
            "d3 P1 & d5 P2": (p1["3"], a5["5"]),
            "d3 P1 & d5 P1": (p1["3"], p1["5"]),
            "d3 P1 & d3 P2": (p1["3"], a3["3"]),
        }
        key = f"day {day_number(day)}"
        shared[key] = {k: len(a & b) for k, (a, b) in pairs.items()}
        names = [f"ibm_fez-{day}-{s}" for s in ("r1", "r2-d5", "r2-d3")]
        cals = [read(folders[n] / "calibration.json") for n in names]
        same[key] = all(c == cals[0] for c in cals)
        created = [
            datetime.fromisoformat(
                read(folders[n] / "job.json")["created_utc"].replace("Z", "+00:00")
            )
            for n in names
        ]
        minutes[key] = (max(created) - min(created)).total_seconds() / 60
    distinct = {}
    for label, suffix, patch in (
        ("P1 d3", "r1", "3"),
        ("P1 d5", "r1", "5"),
        ("P2 d3", "r2-d3", "3"),
        ("P2 d5", "r2-d5", "5"),
    ):
        sets = {frozenset(chips[f"ibm_fez-{d}-{suffix}"][patch]) for d in days}
        distinct[label] = len(sets)
    return {
        "qubits_per_patch": {f"d{p}": len(q) for p, q in chips[f"ibm_fez-{days[0]}-r1"].items()},
        "shared_qubits_by_day": shared,
        "distinct_qubit_sets_over_the_days": distinct,
        "one_calibration_per_day (the three jobs' calibration.json equal)": same,
        "minutes_from_first_to_last_job_of_a_day": minutes,
    }


# --------------------------------------------------------------------------------- blocks
def imhof_sf(x: float, weights, dof) -> float:
    """P(sum_j w_j chi2(dof_j) > x), Imhof's integral."""
    w, h = np.asarray(weights, float), np.asarray(dof, float)
    keep = w > 0
    w, h = w[keep], h[keep]

    def f(u: float) -> float:
        theta = 0.5 * (h * np.arctan(w * u)).sum() - 0.5 * x * u
        rho = np.prod((1 + (w * u) ** 2) ** (h / 4))
        return float(np.sin(theta) / (u * rho))

    val, _ = integrate.quad(f, 0, np.inf, limit=2000, epsabs=1e-13)
    return float(min(max(0.5 + val / np.pi, 0.0), 1.0))


def correlated(T: float) -> dict:
    """p of T = sum z^2 if the two placements' z on each day had correlation rho."""
    grid = np.linspace(0, 1, 101)
    p = np.array([imhof_sf(T, (1 + r, 1 - r), (5, 5)) for r in grid])
    # the integral is good to about 1e-9 absolute; the ends have closed forms
    assert abs(p[0] - stats.chi2.sf(T, 10)) < 1e-8 and abs(p[-1] - stats.chi2.sf(T / 2, 5)) < 1e-8
    p[0], p[-1] = stats.chi2.sf(T, 10), stats.chi2.sf(T / 2, 5)
    return {
        "p_at_rho": {
            "0": float(p[0]),
            "0.5": float(p[50]),
            "51/65": imhof_sf(T, (1 + 51 / 65, 1 - 51 / 65), (5, 5)),
            "1 (T/2 against chi2 5 dof)": float(stats.chi2.sf(T / 2, 5)),
        },
        "max_p_over_rho": float(p.max()),
        "rho_at_max": float(grid[p.argmax()]),
    }


def t_test(x) -> dict:
    x = np.asarray(x, float)
    t = stats.ttest_1samp(x, 0.0)
    return {
        "mean": float(x.mean()),
        "t": float(t.statistic),
        "dof": len(x) - 1,
        "p": float(t.pvalue),
        "p_one_sided (device above)": float(stats.t.sf(t.statistic, len(x) - 1)),
    }


def sign_flip(x) -> dict:
    """Exact randomization test of a zero mean over every sign pattern of the blocks."""
    x = np.asarray(x, float)
    signs = np.array(list(itertools.product((1.0, -1.0), repeat=len(x))))
    m = (signs * x).mean(1)
    tol = 1e-12 * np.abs(x).max()
    return {
        "patterns": len(m),
        "p": float((np.abs(m) >= abs(x.mean()) - tol).mean()),
        "p_one_sided (device above)": float((m >= x.mean() - tol).mean()),
        "smallest_attainable_p": 2 / len(m),
    }


class Hierarchy:
    """ln(O/M) = X beta + a_day + e, Var = diag(v + tau^2) + tau_day^2 [same day], on a grid of
    (tau_day, tau); REML for estimates, ML for the likelihood ratio."""

    def __init__(self, v, day) -> None:
        same = (day[:, None] == day[None, :]).astype(float)
        td, tr = np.meshgrid(TAU, TAU, indexing="ij")
        self.taus = np.stack([td.ravel(), tr.ravel()], 1)
        V = np.diag(v)[None] + self.taus[:, 1, None, None] ** 2 * np.eye(len(v))[None]
        V = V + self.taus[:, 0, None, None] ** 2 * same[None]
        self.Vi = np.linalg.inv(V)
        self.logdet = np.linalg.slogdet(V)[1]

    def project(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """P = Vi - Vi X (X' Vi X)^-1 X' Vi per grid point, with log|X' Vi X| and (X' Vi X)^-1."""
        if X.shape[1] == 0:
            return self.Vi, np.zeros(len(self.Vi)), np.zeros((len(self.Vi), 0, 0))
        ViX = self.Vi @ X
        A = X.T @ ViX
        Ai = np.linalg.inv(A)
        return self.Vi - ViX @ Ai @ ViX.transpose(0, 2, 1), np.linalg.slogdet(A)[1], Ai

    def best(self, Y: np.ndarray, X: np.ndarray, reml: bool) -> tuple[np.ndarray, np.ndarray]:
        """Maximum log-likelihood and its grid index for each row of Y."""
        P, ldA, _ = self.project(X)
        q = np.empty((len(Y), len(P)))
        for c in range(0, len(P), 256):  # chunks keep the intermediate near 40 MB
            q[:, c : c + 256] = np.einsum("gbj,bj->bg", np.matmul(Y[None], P[c : c + 256]), Y)
        ll = -0.5 * (self.logdet[None] + q + (ldA[None] if reml else 0.0))
        i = ll.argmax(1)
        return ll[np.arange(len(Y)), i], i

    def fit(self, y: np.ndarray, X: np.ndarray, dof: int) -> dict:
        _, i = self.best(y[None], X, reml=True)
        g = int(i[0])
        ViX = self.Vi[g] @ X
        Ai = np.linalg.inv(X.T @ ViX)
        beta = Ai @ ViX.T @ y
        se = np.sqrt(np.diag(Ai))
        t = float(stats.t.ppf(0.975, dof))
        return {
            "tau_day": float(self.taus[g, 0]),
            "tau": float(self.taus[g, 1]),
            "mu": [
                {
                    "ratio": float(np.exp(b)),
                    "ci95 (t, days - 1 dof)": [float(np.exp(b - t * s)), float(np.exp(b + t * s))],
                }
                for b, s in zip(beta, se)
            ],
        }

    def lrt(self, y: np.ndarray, X: np.ndarray, rng: np.random.Generator) -> dict:
        """mu = 0 against X beta, ML, the null's variance components kept in the bootstrap."""
        none = np.zeros((len(y), 0))
        l1, _ = self.best(y[None], X, reml=False)
        l0, i0 = self.best(y[None], none, reml=False)
        lr = float(2 * (l1[0] - l0[0]))
        g = int(i0[0])
        cov = np.linalg.inv(self.Vi[g])
        Y = rng.multivariate_normal(np.zeros(len(y)), cov, size=B)
        lr_b = 2 * (self.best(Y, X, reml=False)[0] - self.best(Y, none, reml=False)[0])
        return {
            "LR": lr,
            "p_bootstrap": float((1 + (lr_b >= lr - 1e-12).sum()) / (B + 1)),
            "p_chi2 (asymptotic)": float(stats.chi2.sf(lr, X.shape[1])),
            "null_tau_day": float(self.taus[g, 0]),
            "null_tau": float(self.taus[g, 1]),
            "replicates": B,
        }


def bootstrap_days(x) -> np.ndarray:
    """Means over every resample of the blocks with replacement (all n^n, equally weighted)."""
    x = np.asarray(x, float)
    idx = np.array(list(itertools.product(range(len(x)), repeat=len(x))))
    return x[idx].mean(1)


def blocks(D: ES.Data, rng: np.random.Generator) -> dict:
    out: dict = {}
    day, pl = D.day, D.placement
    ones = np.ones((10, 1))
    each = np.stack([pl == 1, pl == 2], 1).astype(float)
    for fit in FITS:
        out[fit] = {}
        for q in X_TESTS:
            s = D.terms(fit, q)
            z = (s.O - s.M) / s.half
            y = np.log(s.O / s.M)
            v = (s.sO / s.O) ** 2 + (s.sM / s.M) ** 2
            T = float((z**2).sum())
            if fit == "wls":
                assert abs(T - D.combined["model"][q]["chi2"]) < 1e-9
            per = {}
            for p in PLACEMENTS:
                zp = z[pl == p]
                per[f"placement {p}"] = {
                    **ES.chi2_p((zp**2).sum(), 5),
                    "offset": ES.chi2_p(5 * zp.mean() ** 2, 1),
                    "scatter": ES.chi2_p(((zp - zp.mean()) ** 2).sum(), 4),
                }
            per["smaller p times 2 (either placement)"] = min(
                1.0, 2 * min(per[f"placement {p}"]["p"] for p in PLACEMENTS)
            )
            H = Hierarchy(v, day)
            yd = day_means(y, day)
            boot = np.exp(bootstrap_days(yd))
            out[fit][q] = {
                "prespecified (10 dof)": ES.chi2_p(T, 10),
                "correlated within a day": correlated(T),
                "per placement (5 dof)": per,
                "day blocks": {
                    "z": {**t_test(day_means(z, day)), "sign_flip": sign_flip(day_means(z, day))},
                    "ln(O/M)": {**t_test(yd), "sign_flip": sign_flip(yd)},
                },
                "hierarchical": {
                    "one mu": {**H.fit(y, ones, 4), "test mu = 0": H.lrt(y, ones, rng)},
                    "mu per placement": {
                        **H.fit(y, each, 4),
                        "test both mu = 0": H.lrt(y, each, rng),
                    },
                },
                "O/M": {
                    **{f"placement {p} (t, 4 dof)": ES.t_interval(y[pl == p]) for p in PLACEMENTS},
                    "day blocks (t, 4 dof)": ES.t_interval(yd),
                    "day-block bootstrap (percentile, all 3125)": {
                        "ratio": float(np.exp(yd.mean())),
                        "ci95": floats(np.percentile(boot, [2.5, 97.5])),
                    },
                    "per_repetition": floats(s.O / s.M),
                },
            }
    return out


# --------------------------------------------------------------------------------- counts
def slope(j: dict, b: str, p: str) -> np.ndarray:
    """dP/df: the line through the f = 0 and f = 1 predictions."""
    return np.array(j["predicted"][F1]["P"][b][p]) - np.array(j["predicted"][F0]["P"][b][p])


def binomial_nll(k, N, P) -> float:
    P = np.clip(P, 1e-12, 1 - 1e-12)
    return float(-(k * np.log(P) + (N - k) * np.log1p(-P)).sum())


def anchored_fit(k, N, Pm, n) -> dict:
    """Binomial fit of P(n) = [1 - a r^(n-2) F(n)] / 2, F = 1 - 2 P_model, on rounds n >= 2; the
    95% profile interval of r with the quasi-likelihood widening phi = max(1, Pearson / dof).
    Round 1 against that trend: its excess over the trend's value at n = 1, and the likelihood
    ratio (1 dof, divided by phi) of round 1 left free against round 1 on the trend."""
    k, n = np.asarray(k, float), np.asarray(n, float)
    F, ex = 1 - 2 * np.asarray(Pm, float), n - 2
    late = n >= 2

    def curve(x, m=late) -> np.ndarray:
        return (1 - np.exp(x[0] + x[1] * ex[m]) * F[m]) / 2

    def nll(x, m=late) -> float:
        return binomial_nll(k[m], N, curve(x, m))

    def fit(m) -> tuple[np.ndarray, float]:
        opt = {"xatol": 1e-10, "fatol": 1e-12}
        res = minimize(nll, np.zeros(2), args=(m,), method="Nelder-Mead", options=opt)
        return res.x, float(res.fun)

    (la, lr), best = fit(late)
    _, on_trend = fit(np.ones(len(k), bool))
    P = curve((la, lr))
    dof = int(late.sum()) - 2
    pearson = float(((k[late] - N * P) ** 2 / (N * P * (1 - P))).sum() / dof)
    phi = max(pearson, 1.0)

    def prof(x: float) -> float:
        return minimize_scalar(lambda a: nll((a, x)), bounds=(-3, 3), method="bounded").fun

    def edge(sign: int) -> float | None:
        f = lambda x: 2 * (prof(x) - best) / phi - stats.chi2.ppf(0.95, 1)  # noqa: E731
        far = lr + sign * 2.0
        return None if f(far) < 0 else brentq(f, *sorted((lr, far)), xtol=1e-9)

    lo, hi = edge(-1), edge(+1)
    Fl, nl = F[late], n[late]
    q = (Fl[-1] / Fl[0]) ** (1 / (nl[-1] - nl[0])) if Fl[-1] > 0 else float("nan")
    first = ~late
    trend1 = float(curve((la, lr), first)[0])
    free1 = binomial_nll(k[first], N, k[first] / N)  # round 1 fitted exactly
    lr_round1 = 2 * (on_trend - (best + free1)) / phi
    return {
        "a": float(np.exp(la)),
        "a0 (a / r^2, at n = 0)": float(np.exp(la - 2 * lr)),
        "r": float(np.exp(lr)),
        "r_ci95": [None if x is None else float(np.exp(x)) for x in (lo, hi)],
        "deviance_dof": dof,
        "pearson_per_dof": pearson,
        "model_fidelity_per_round_q": float(q),
        "eps_model": float((1 - q) / 2),
        "eps_device": float((1 - np.exp(lr) * q) / 2),
        "round 1 against the trend": {
            "device_minus_trend_pp": float(100 * (k[first][0] / N - trend1)),
            "LR (1 dof, over phi)": float(lr_round1),
            "p": float(stats.chi2.sf(max(lr_round1, 0.0), 1)),
        },
    }


def snr_10pct(A: float, eps: float, n, P, N: int) -> np.ndarray:
    """How far P(n) moves when eps grows by 10%, in the chip's standard errors."""
    n = np.asarray(n, float)
    dP = A * n * (1 - 2 * eps) ** (n - 1) * 0.1 * eps
    return dP / np.sqrt(np.asarray(P) * (1 - np.asarray(P)) / N)


def mean_ci(x) -> dict:
    """Mean over the days with its t 95% interval (the days as blocks)."""
    x = np.asarray(x, float)
    t, se = stats.t.ppf(0.975, len(x) - 1), x.std(ddof=1) / sqrt(len(x))
    return {"mean": float(x.mean()), "ci95": [float(x.mean() - t * se), float(x.mean() + t * se)]}


def chi2_parts(r, S, P, late) -> dict:
    """r' S^-1 r over a curve's points, round 1 and rounds 2-8 apart, and without the points
    near 1/2."""
    far = P < NEAR_HALF
    parts = {
        "all": np.ones(len(r), bool),
        "round 1": ~late,
        "rounds 2-8": late,
        "P_model < 0.45": far,
        "rounds 2-8, P_model < 0.45": late & far,
    }
    out = {}
    for name, m in parts.items():
        x = float(r[m] @ np.linalg.solve(S[np.ix_(m, m)], r[m]))
        out[name] = {"chi2": x, "dof": int(m.sum()), "p": float(stats.chi2.sf(x, m.sum()))}
    return out


def counts(D: ES.Data) -> dict:
    jobs, preds = D.counts["jobs"], D.fits["predictions"]
    rounds = np.array(D.counts["rounds"], float)
    late = rounds >= 2
    variants = {"prespecified": {n: j["f_uncertainty"] for n, j in jobs.items()}}
    try:
        boot = result("sensitivity.json")["sigma_f"]["per_job"]
        variants["idle_noise_only"] = {n: v["sigma_f_idle_noise"] for n, v in boot.items()}
    except (FileNotFoundError, KeyError):
        pass
    variants["none"] = dict.fromkeys(jobs, 0.0)
    chi: dict = {label: {} for label in variants}
    joint: dict = {label: {} for label in variants}  # a job's curves in one basis, jointly
    curves: dict = {}
    for name, j in jobs.items():
        N, Ns = j["shots"], j["predicted"][PRED]["shots"]
        rep = f"{j['day']} r{j['rep']}"
        for b in "XZ":
            k = np.concatenate([j["observed"][b][p] for p in j["patches"]]).astype(float)
            P = np.concatenate([j["predicted"][PRED]["P"][b][p] for p in j["patches"]])
            g = np.concatenate([slope(j, b, p) for p in j["patches"]])
            r = k / N - P
            for label, sf in variants.items():
                S = np.diag(P * (1 - P) * (1 / N + 1 / Ns)) + sf[name] ** 2 * np.outer(g, g)
                key = (rep, b)
                joint[label][key] = joint[label].get(key, 0.0) + float(r @ np.linalg.solve(S, r))
                for i, p in enumerate(j["patches"]):
                    c = slice(6 * i, 6 * i + 6)
                    chi[label][f"{rep} {b} d{p}"] = chi2_parts(r[c], S[c, c], P[c], late)
            for i, p in enumerate(j["patches"]):
                c = slice(6 * i, 6 * i + 6)
                fit = preds[name][PRED][b][p]["binomial"]
                sf = variants["prespecified"][name]
                curves[f"{rep} {b} d{p}"] = {
                    "job": name,
                    "placement": j["rep"],
                    "day": day_number(j["day"]),
                    "k": floats(k[c]),
                    "N": N,
                    "P_device": floats(k[c] / N),
                    "P_model": floats(P[c]),
                    "excess_pp": floats(100 * r[c]),
                    "sd_pp [chip, simulator, f term]": [
                        floats(100 * np.sqrt(P[c] * (1 - P[c]) / N)),
                        floats(100 * np.sqrt(P[c] * (1 - P[c]) / Ns)),
                        floats(100 * np.abs(g[c]) * sf),
                    ],
                    "near_half (P_model >= 0.45)": [bool(x) for x in P[c] >= NEAR_HALF],
                    "device_at_or_above_half": [bool(x) for x in k[c] >= N / 2],
                    "snr_10pct": floats(snr_10pct(fit["A"], fit["eps"], rounds, P[c], N)),
                    "chi2 (prespecified sigma_f)": chi["prespecified"][f"{rep} {b} d{p}"],
                    "anchored (rounds 2-8)": anchored_fit(k[c], N, P[c], rounds),
                }
    groups: dict = {}
    for b in "XZ":
        for pl in PLACEMENTS:
            for d in "35":
                keys = [
                    ck
                    for ck, v in curves.items()
                    if ck.endswith(f"{b} d{d}") and v["placement"] == pl
                ]
                assert len(keys) == 5  # one curve a day, each from its own job
                rows = [curves[ck] for ck in keys]
                sums: dict = {}
                for label in variants:
                    sums[label] = {}
                    for part in chi[label][keys[0]]:
                        x = sum(chi[label][ck][part]["chi2"] for ck in keys)
                        dof = sum(chi[label][ck][part]["dof"] for ck in keys)
                        sums[label][part] = {
                            "chi2": x,
                            "dof": dof,
                            "p": float(stats.chi2.sf(x, dof)) if dof else None,
                        }
                ex = np.array([r["excess_pp"] for r in rows])  # days x round counts
                by_n = {
                    str(int(n)): {
                        **mean_ci(ex[:, i]),
                        "days_device_above": int((ex[:, i] > 0).sum()),
                        "P_model_range": [
                            float(min(r["P_model"][i] for r in rows)),
                            float(max(r["P_model"][i] for r in rows)),
                        ],
                        "near_half_days": int(
                            sum(r["near_half (P_model >= 0.45)"][i] for r in rows)
                        ),
                        "snr_10pct_median": float(np.median([r["snr_10pct"][i] for r in rows])),
                    }
                    for i, n in enumerate(rounds)
                }
                an = [r["anchored (rounds 2-8)"] for r in rows]
                ratio = np.array([a["eps_device"] / a["eps_model"] for a in an])
                ok = np.isfinite(ratio) & (ratio > 0)
                lr1 = sum(a["round 1 against the trend"]["LR (1 dof, over phi)"] for a in an)
                groups[f"{b} placement {pl} d{d}"] = {
                    "chi2 (sum over the five days)": sums,
                    "excess_pp by round count (mean over days, t 95%)": by_n,
                    "anchored (rounds 2-8)": {
                        "r_by_day": [a["r"] for a in an],
                        "r_ci95_by_day": [a["r_ci95"] for a in an],
                        "a (fidelity ratio at round 2) by day": [a["a"] for a in an],
                        "a0 (at n = 0: preparation and final readout; t over days)": ES.t_interval(
                            np.log([a["a0 (a / r^2, at n = 0)"] for a in an])
                        ),
                        "eps_device_over_model_by_day": floats(ratio),
                        "eps_device_over_model (t over days)": ES.t_interval(np.log(ratio[ok])),
                        "days_used": int(ok.sum()),
                        "excess_eps_per_round_pp": mean_ci(
                            [100 * (a["eps_device"] - a["eps_model"]) for a in an]
                        ),
                        "pearson_per_dof_max": max(a["pearson_per_dof"] for a in an),
                        "round 1 against the trend": {
                            "device_minus_trend_pp": mean_ci(
                                [
                                    a["round 1 against the trend"]["device_minus_trend_pp"]
                                    for a in an
                                ]
                            ),
                            "LR summed over days (5 dof)": float(lr1),
                            "p": float(stats.chi2.sf(lr1, 5)),
                        },
                    },
                }
            sums = {}
            for label in variants:
                if pl == 1:  # one job a day: the two distances with their shared f term
                    x = sum(
                        v for (rep, bb), v in joint[label].items() if bb == b and rep.endswith("r1")
                    )
                else:  # two jobs a day, independent
                    x = sum(
                        groups[f"{b} placement 2 d{d}"]["chi2 (sum over the five days)"][label][
                            "all"
                        ]["chi2"]
                        for d in "35"
                    )
                sums[label] = {"chi2": float(x), "dof": 60, "p": float(stats.chi2.sf(x, 60))}
            groups[f"{b} placement {pl}, both distances"] = sums
    tally = {}
    for b in "XZ":
        for d in "35":
            rows = [v for ck, v in curves.items() if ck.endswith(f"{b} d{d}")]
            tally[f"{b} d{d}"] = {
                "near_half curves by round count (of 10)": {
                    str(int(n)): int(sum(r["near_half (P_model >= 0.45)"][i] for r in rows))
                    for i, n in enumerate(rounds)
                },
                "points with the device at or above 1/2 (of 60)": int(
                    sum(sum(r["device_at_or_above_half"]) for r in rows)
                ),
                "points with the device above the model (of 60)": int(
                    sum(sum(x > 0 for x in r["excess_pp"]) for r in rows)
                ),
            }
    return {
        "what": "k(n) of N against the model's P(n) at the measured f; chi2 with the chip's and the "
        "simulator's binomial variance at the model's P plus sigma_f^2 g g^T, g = P(f = 1) - "
        "P(f = 0); 'none' leaves out the f term. Groups: sums over the five days, one job each",
        "tally": tally,
        "groups": groups,
        "curves": curves,
    }


# ---------------------------------------------------------------------------- equivalence
def tost(x, margin: float) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    m, se = float(x.mean()), float(x.std(ddof=1) / sqrt(n))
    t90 = stats.t.ppf(0.95, n - 1)
    lo, hi = m - t90 * se, m + t90 * se
    return {
        "p_tost": float(
            max(stats.t.sf((m + margin) / se, n - 1), stats.t.cdf((m - margin) / se, n - 1))
        ),
        "equivalent": bool(-margin < lo and hi < margin),
    }


def interval(x) -> dict:
    x = np.asarray(x, float)
    n = len(x)
    m, se = float(x.mean()), float(x.std(ddof=1) / sqrt(n))
    t90, t95 = stats.t.ppf(0.95, n - 1), stats.t.ppf(0.975, n - 1)
    return {
        "mean": m,
        "sd": float(x.std(ddof=1)),
        "se": se,
        "dof": n - 1,
        "ci90": [m - t90 * se, m + t90 * se],
        "ci95": [m - t95 * se, m + t95 * se],
        "smallest_margin_certified": float(max(abs(m - t90 * se), abs(m + t90 * se))),
        "differs_from_zero (95%)": bool(m - t95 * se > 0 or m + t95 * se < 0),
    }


def certify_factor(n: int, rng: np.random.Generator) -> float:
    """c with P(certifying a margin c sd) = 0.8 when the truth is zero: n units, t interval."""
    x = rng.standard_normal((MC, n))
    reach = np.abs(x.mean(1)) + stats.t.ppf(0.95, n - 1) * x.std(1, ddof=1) / sqrt(n)
    return float(np.quantile(reach, 0.8))


def t_power(offset: float, sd: float, n: int) -> float:
    """Power of the two-sided one-sample t test at 5% against a true mean offset."""
    t, nc = stats.t.ppf(0.975, n - 1), offset / (sd / sqrt(n))
    return float(stats.nct.sf(t, n - 1, nc) + stats.nct.cdf(-t, n - 1, nc))


def power_certify(sd: float, n: int, margin: float, rng: np.random.Generator) -> float:
    x = sd * rng.standard_normal((MC, n))
    reach = np.abs(x.mean(1)) + stats.t.ppf(0.95, n - 1) * x.std(1, ddof=1) / sqrt(n)
    return float((reach < margin).mean())


def equivalence(D: ES.Data, rng: np.random.Generator) -> dict:
    c5 = certify_factor(5, rng)
    out: dict = {
        "what": "Lambda_X, O - M. Equivalent within a margin when the 90% t interval lies inside it "
        "(two one-sided tests at 5%); certified margin: the larger end of that interval. Power: "
        "the chance of certifying the margin if the model were exactly right, five units, at the "
        "observed sd between them and at the stated errors alone",
        "margin_certified_80pct_in_sd_units (5 units, t 4 dof)": c5,
        "margin_certified_80pct_in_se_units (known errors)": float(
            stats.norm.ppf(0.95) + stats.norm.ppf(0.9)
        ),
    }
    try:
        sysw = result("sensitivity.json")["windows"]["shifts"]
    except (FileNotFoundError, KeyError):
        sysw = None
    for fit in FITS:
        s = D.terms(fit, "X lambda")
        d, half, pl = s.O - s.M, s.half, D.placement
        units = {f"placement {p}": (d[pl == p], half[pl == p], s.M[pl == p]) for p in PLACEMENTS}
        hd = np.sqrt(np.array([(half[D.day == k] ** 2).sum() for k in np.unique(D.day)])) / 2
        units["day blocks"] = (day_means(d, D.day), hd, s.M)
        blk: dict = {}
        for label, (x, sig, M) in units.items():
            stated = float(np.sqrt((sig**2).mean()))
            w = 1 / sig**2
            mw = float((w * x).sum() / w.sum())
            sew = float(1 / sqrt(w.sum()))
            q = float((w * (x - mw) ** 2).sum())
            birge = sew * sqrt(max(q / (len(x) - 1), 1.0))
            z90 = stats.norm.ppf(0.95)
            blk[label] = {
                "per_unit": floats(x),
                **interval(x),
                **{f"margin {m}": tost(x, m) for m in MARGINS},
                "stated errors (inverse variance)": {
                    "mean": mw,
                    "se": sew,
                    "ci90": [mw - z90 * sew, mw + z90 * sew],
                    "chi2_about_mean": q,
                    "dof": len(x) - 1,
                    "se_widened_by_spread": birge,
                    "ci90_widened (t)": [
                        mw - stats.t.ppf(0.95, len(x) - 1) * birge,
                        mw + stats.t.ppf(0.95, len(x) - 1) * birge,
                    ],
                },
                "rms_stated_sigma": stated,
                "power_to_certify_0.05": {
                    "at the observed sd": power_certify(float(x.std(ddof=1)), len(x), 0.05, rng),
                    "at the stated errors (sd = rms sigma)": power_certify(
                        stated, len(x), 0.05, rng
                    ),
                },
                "margin_certified_with_80pct": {
                    "at the observed sd": c5 * float(x.std(ddof=1)),
                    "at the stated errors": c5 * stated,
                },
                "power_to_detect_offset (t test, 5%, observed sd)": {
                    f"{o:.2f}": t_power(o, float(x.std(ddof=1)), len(x)) for o in (0.05, 0.10)
                },
                "predicted_mean": float(M.mean()),
            }
        out[fit] = blk
    px = D.combined["pooled_x"]
    diff = px["lambda"] - px["predicted"]
    t9 = stats.t.ppf(0.95, 9)
    out["prespecified pool (WL, inverse variance widened by spread, t 9 dof)"] = {
        "observed": px["lambda"],
        "se": px["lambda_uncertainty"],
        "predicted": px["predicted"],
        "difference": diff,
        "ci90": [diff - t9 * px["lambda_uncertainty"], diff + t9 * px["lambda_uncertainty"]],
        "ci95": [
            diff - stats.t.ppf(0.975, 9) * px["lambda_uncertainty"],
            diff + stats.t.ppf(0.975, 9) * px["lambda_uncertainty"],
        ],
        "note": "the ten repetitions as independent; the prediction's own error left out, as in the rule",
    }
    if sysw:
        out["systematic of O - M from windows and fits (sensitivity.py, half range)"] = {
            label: v["device_minus_model"]["systematic (half range without 3-8)"]
            for label, v in sysw.items()
        }
    return out


# ------------------------------------------------------------------------------- headline
def headline(res: dict) -> dict:
    bl, cn, eq = res["blocks"], res["counts"]["groups"], res["equivalence"]

    def ci(v: dict) -> list:
        return [v["ratio"], *v["ci95"]]

    tests, ratios = {}, {}
    for q in X_TESTS:
        w = bl["wls"][q]
        tests[q] = {
            "prespecified (chi2, 10 dof)": w["prespecified (10 dof)"]["p"],
            "two placements' z on a day fully correlated (max over rho)": w[
                "correlated within a day"
            ]["max_p_over_rho"],
            "placement 1 alone (5 dof)": w["per placement (5 dof)"]["placement 1"]["p"],
            "placement 2 alone (5 dof)": w["per placement (5 dof)"]["placement 2"]["p"],
            "offset, day-block t on z (4 dof)": w["day blocks"]["z"]["p"],
            "offset, day-block sign flip (smallest possible 0.0625)": w["day blocks"]["z"][
                "sign_flip"
            ]["p"],
            "offset, hierarchical LRT (bootstrap)": w["hierarchical"]["one mu"]["test mu = 0"][
                "p_bootstrap"
            ],
        }
        b = bl["binomial"][q]
        ratios[q] = {
            **{f"placement {p}": ci(b["O/M"][f"placement {p} (t, 4 dof)"]) for p in PLACEMENTS},
            "day blocks": ci(b["O/M"]["day blocks (t, 4 dof)"]),
            "hierarchical": [
                b["hierarchical"]["one mu"]["mu"][0]["ratio"],
                *b["hierarchical"]["one mu"]["mu"][0]["ci95 (t, days - 1 dof)"],
            ],
            "tau_day, tau (REML)": [
                b["hierarchical"]["one mu"]["tau_day"],
                b["hierarchical"]["one mu"]["tau"],
            ],
        }
    raw: dict = {}
    for b in "XZ":
        for pl in PLACEMENTS:
            for d in "35":
                g = cn[f"{b} placement {pl} d{d}"]
                s = g["chi2 (sum over the five days)"]["prespecified"]
                ex = g["excess_pp by round count (mean over days, t 95%)"]
                an = g["anchored (rounds 2-8)"]
                raw[f"{b} P{pl} d{d}"] = {
                    "chi2/dof [all, round 1, rounds 2-8, P_model < 0.45]": [
                        [s[k]["chi2"], s[k]["dof"]]
                        for k in ("all", "round 1", "rounds 2-8", "P_model < 0.45")
                    ],
                    "excess_pp by n": {n: round(v["mean"], 2) for n, v in ex.items()},
                    "eps after round 1, device / model [ratio, 95% low, high]": ci(
                        an["eps_device_over_model (t over days)"]
                    ),
                    "excess eps per round after round 1 (pp)": an["excess_eps_per_round_pp"][
                        "mean"
                    ],
                    "fidelity ratio at n = 0 [ratio, 95% low, high]": ci(
                        an["a0 (at n = 0: preparation and final readout; t over days)"]
                    ),
                    "round 1 above the trend (pp)": an["round 1 against the trend"][
                        "device_minus_trend_pp"
                    ]["mean"],
                }
    lam = {}
    for fit in FITS:
        for k, v in eq[fit].items():
            lam[f"{fit}, {k}"] = {
                "mean": v["mean"],
                "ci90": v["ci90"],
                "certified_margin": v["smallest_margin_certified"],
                "within 0.05": v["margin 0.05"]["equivalent"],
                "within 0.10": v["margin 0.1"]["equivalent"],
                "chance to certify 0.05 if exact [observed sd, stated errors]": [
                    v["power_to_certify_0.05"]["at the observed sd"],
                    v["power_to_certify_0.05"]["at the stated errors (sd = rms sigma)"],
                ],
                "margin certifiable at 80% [observed sd, stated errors]": [
                    v["margin_certified_with_80pct"]["at the observed sd"],
                    v["margin_certified_with_80pct"]["at the stated errors"],
                ],
            }
    pool = eq["prespecified pool (WL, inverse variance widened by spread, t 9 dof)"]
    lam["prespecified pool (WL)"] = {"mean": pool["difference"], "ci90": pool["ci90"]}
    return {
        "X tests (WL), p": tests,
        "O/M (binomial) [ratio, 95% low, high]": ratios,
        "raw counts against the model (prespecified sigma_f)": raw,
        "Lambda_X O - M": lam,
    }


def main() -> None:
    rng = np.random.default_rng(SEED)
    D = ES.Data()
    res = {
        "structure": structure(),
        "blocks": blocks(D, rng),
        "counts": counts(D),
        "equivalence": equivalence(D, rng),
    }
    res = {"headline": headline(res), **res}
    h = res["headline"]
    for q, v in h["X tests (WL), p"].items():
        print(f"{q:9s} " + "  ".join(f"{p:.2g}" for p in v.values()))
    for c, v in h["raw counts against the model (prespecified sigma_f)"].items():
        r = v["eps after round 1, device / model [ratio, 95% low, high]"]
        print(
            f"{c:9s} chi2 {v['chi2/dof [all, round 1, rounds 2-8, P_model < 0.45]'][0][0]:7.0f}/30, "
            f"eps after round 1 x{r[0]:.2f} ({r[1]:.2f}-{r[2]:.2f})"
        )
    for k, v in h["Lambda_X O - M"].items():
        print(
            f"Lambda_X O - M {k:22s} {v['mean']:+.3f}, 90% [{v['ci90'][0]:+.3f}, {v['ci90'][1]:+.3f}]"
        )
    write_json("dependence.json", res)


if __name__ == "__main__":
    main()
