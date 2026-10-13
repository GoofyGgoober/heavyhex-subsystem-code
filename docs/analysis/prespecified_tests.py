"""The tests fixed before the run, under both choices of sigma_f, with the pooled verdict, the
pairs' table and what the tests could detect (Sec. tests, Sec. results, the tables of rules and
of pairs, Fig. lambda (b), App. stats).

A pair-day is one pair on one day (the software's "rep"). Pair 1 ran both patches in one job,
pair 2 in two. Nothing is simulated or refitted: everything comes from runs/combined.json, the
jobs' idle tests and the stored results of fits.py and sensitivity.py, and is checked against
them where they overlap.

  per_pair_day  O, sigma_O, M and sigma_M of the six tests and z = (O - M) / sqrt(sigma_O^2 +
                sigma_M^2), with sigma_f as prespecified (1.25 s_f / sqrt(n), the spread of the
                qubits) and, in X memory, from the idle test's noise alone (sensitivity.py's
                parametric bootstrap); the prespecified verdict, worse if Lambda_X + 1.645 sigma_O < 1
  chi2          sum z^2 over the ten pair-days (bound 18.31) and over each pair's five (11.07), p,
                Holm p over the three X tests, |z| < 2 counts, offset 10 zbar^2 (1 dof) and scatter
                (9 dof); both fits and both sigma_f. Z memory as the prespecified software computed
                it, outside the decision (its f term is negligible)
  pool          Eq. (pool) recomputed and checked against combined.json: the Birge-widened error,
                t_{9,0.95}, the one-sided bound and the verdict; the prediction pooled with the same
                weights; each pair's five pair-days pooled alike; the binomial fit pooled alike
  sigma_f       both sigma_f per job, and how far the per-qubit idle fits (run._fit_share)
                scatter beyond their shot noise: each qubit's weighted residual chi2 about its line,
                and the uniformity chi2/nu (run._uniformity) with each qubit's error scaled by
                sqrt(max(1, reduced chi2))
  windows       the fit-window systematic on each pair's mean Lambda_X: half the range over five
                windows of rounds and both fits, rounds 3-8 left out (sensitivity.py); and what
                leaving out round 1 does
  table_pairs   the pairs' table: plain means over the five days, error sqrt(sum sigma_i^2) / 5
                times sqrt(chi2_0 / 4) when that exceeds one, chi2_0 about the plain mean, for the
                observations (each fit's own sigma_i, as effect_sizes.py) and for the predictions
                (sigma_M) alike; Lambda_X with its fit-window systematic
  power         against the same Lambda_X miss of 0.06 in every pair-day: one pair-day's chance of
                |z| >= 2, a pair's chi2 (5 dof) and the ten pair-days' chi2 (10 dof), with sigma_f
                as prespecified, from the idle test's noise and without the f term; the relative
                miss in eps found with 80% probability; beside the design study (power.json)

    python docs/analysis/prespecified_tests.py      # seconds; writes results/prespecified_tests.json
"""

from __future__ import annotations

from math import sqrt

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from heavyhex.experiment import run as R

import effect_sizes as ES
import fits as FT
from common import RUNS, day_number, read, result, write_json

TESTS = ("X lambda", "X eps3", "X eps5", "Z lambda", "Z eps3", "Z eps5")
X_TESTS = TESTS[:3]
FITS = ("wls", "binomial")  # wls: the prespecified (log-linear) fit
SIGMA_F = ("prespecified", "idle_noise_only", "no_f_term")
PAIRS = (1, 2)
MISS = 0.06  # the Lambda_X miss the design study (docs/power.py) was sized to catch
ONE_SIDED = 1.645  # the per-pair-day verdict's critical value (run.verdict)


def floats(x) -> list:
    return [float(v) for v in x]


def chi2_p(x: float, dof: int) -> dict:
    return {"chi2": float(x), "dof": int(dof), "p": float(stats.chi2.sf(x, dof))}


# ----------------------------------------------------------------------------------- data
class Data:
    def __init__(self) -> None:
        self.combined = read(RUNS / "combined.json")
        self.reps = self.combined["reps"]  # day, then pair
        self.keys = [f"{r['day']} r{r['rep']}" for r in self.reps]
        self.day = np.array([day_number(r["day"]) for r in self.reps])
        self.pair = np.array([r["rep"] for r in self.reps])
        fits = result("fits.json")
        self.per_rep = fits["per_rep"]
        self.curves = fits["curves"]
        self.tests = {
            "wls": [r["tests"] for r in self.reps],  # the prespecified analysis itself
            "binomial": [fits["agreement_tests"]["binomial"][k] for k in self.keys],
        }
        self.sens = result("sensitivity.json")
        self.jobs = self.sens["sigma_f"]["per_job"]  # sigma_f both ways, per job
        for t in self.tests["wls"]:  # IBM did not recalibrate during any job
            assert all(t[q]["calibration_gap"] == 0 for q in TESTS)

    def job(self, i: int, d: str) -> str:
        """The job that ran pair-day i's distance-d patch."""
        folders = self.reps[i]["folders"]
        return folders[0] if len(folders) == 1 else next(f for f in folders if f.endswith(f"d{d}"))

    def mask(self, p: int) -> np.ndarray:
        return self.pair == p


def scale_of(D: Data, label: str, job: str) -> float:
    """The factor on a job's f term: 1 as prespecified, the noise-only ratio, or 0."""
    return {"prespecified": 1.0, "no_f_term": 0.0}.get(label, D.jobs[job]["ratio"])


def x_rows(D: Data, fit: str, label: str) -> list[dict]:
    """Each pair-day's X tests with every job's f term scaled to the chosen sigma_f.

    A split pair-day's Lambda adds its two jobs' relative terms in quadrature (run._ratio_test,
    fits.ratio_test); a paired one scales Lambda's own term (as sensitivity.scaled).
    """
    out = []
    for i, t in enumerate(D.tests[fit]):
        j3, j5 = D.job(i, "3"), D.job(i, "5")
        e3 = {**t["X eps3"], "from_f": t["X eps3"]["from_f"] * scale_of(D, label, j3)}
        e5 = {**t["X eps5"], "from_f": t["X eps5"]["from_f"] * scale_of(D, label, j5)}
        lam = dict(t["X lambda"])
        if j3 == j5:
            lam["from_f"] *= scale_of(D, label, j3)
        else:
            lam["from_f"] = FT.ratio_test(e3, e5)["from_f"]
        out.append({"X eps3": e3, "X eps5": e5, "X lambda": lam})
    return out


def sigma_M(t: dict) -> float:
    """Eq. (bound): simulator noise, the f term and the (zero) recalibration term."""
    return sqrt(t["simulator_noise"] ** 2 + t["from_f"] ** 2 + t["calibration_gap"] ** 2 / 2)


def z_of(t: dict) -> float:
    return (t["observed"] - t["predicted"]) / sqrt(t["observed_uncertainty"] ** 2 + sigma_M(t) ** 2)


# --------------------------------------------------------------------------- per pair-day
def per_pair_day(D: Data) -> list[dict]:
    noise = {fit: x_rows(D, fit, "idle_noise_only") for fit in FITS}
    rows = []
    for i in range(len(D.reps)):
        lam = D.tests["wls"][i]["X lambda"]
        upper = lam["observed"] + ONE_SIDED * lam["observed_uncertainty"]
        assert (upper < 1) == (D.reps[i]["verdict"] == "d = 5 worse")
        jobs = dict.fromkeys(D.job(i, d) for d in "35")
        row: dict = {
            "day": int(D.day[i]),
            "pair": int(D.pair[i]),
            "pairing": D.reps[i]["pairing"],
            "jobs": {
                j: {k: D.jobs[j][k] for k in ("f", "sigma_f_prespecified", "sigma_f_idle_noise")}
                for j in jobs
            },
            "verdict (WL)": {
                "Lambda_X": lam["observed"],
                "sigma_O": lam["observed_uncertainty"],
                "Lambda_X + 1.645 sigma_O": upper,
                "verdict": D.reps[i]["verdict"],
            },
        }
        for fit in FITS:
            blk = {}
            for q in TESTS:
                t = D.tests[fit][i][q]
                cell = {
                    "O": t["observed"],
                    "sigma_O": t["observed_uncertainty"],
                    "M": t["predicted"],
                    "sigma_M": sigma_M(t),
                    "z": z_of(t),
                }
                if q in X_TESTS:
                    tn = noise[fit][i][q]
                    cell["sigma_M_idle_noise_only"] = sigma_M(tn)
                    cell["z_idle_noise_only"] = z_of(tn)
                blk[q] = cell
            row[fit] = blk
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------------- chi2
def summarize(D: Data, z) -> dict:
    z = np.asarray(z, float)
    zbar = float(z.mean())
    return {
        **chi2_p((z**2).sum(), len(z)),
        "abs_z_below_2": int((np.abs(z) < 2).sum()),
        "z_positive": int((z > 0).sum()),
        "mean_z": zbar,
        "offset": chi2_p(len(z) * zbar**2, 1),
        "scatter": chi2_p(((z - zbar) ** 2).sum(), len(z) - 1),
        "per_pair": {
            f"pair {p}": {
                **chi2_p((z[D.mask(p)] ** 2).sum(), int(D.mask(p).sum())),
                "abs_z_below_2": int((np.abs(z[D.mask(p)]) < 2).sum()),
            }
            for p in PAIRS
        },
        "z": floats(z),
    }


def chi2_tests(D: Data) -> dict:
    out: dict = {
        "what": "chi2 = sum z^2, bound chi2_{nu,0.95}: 18.31 for the ten pair-days, 11.07 for a "
        "pair's five. sigma_f: prespecified = 1.25 s_f / sqrt(n); idle_noise_only = the idle "
        "test's parametric bootstrap (sensitivity.py); no_f_term = sigma_M from the simulator's "
        "noise alone. The decision is the WL fit with the prespecified sigma_f; the rest was "
        "chosen after the run.",
        "bounds": {
            "10 dof": float(stats.chi2.ppf(0.95, 10)),
            "5 dof": float(stats.chi2.ppf(0.95, 5)),
        },
    }
    for fit in FITS:
        out[fit] = {}
        for label in SIGMA_F:
            rows = x_rows(D, fit, label)
            blk = {q: summarize(D, [z_of(r[q]) for r in rows]) for q in X_TESTS}
            for q, p in zip(X_TESTS, ES.holm([blk[q]["p"] for q in X_TESTS])):
                blk[q]["p_holm_over_the_three_X_tests"] = p
            out[fit][label] = blk
            # the check: sensitivity.py computed the same, the prespecified run the first
            for q in X_TESTS:
                assert abs(blk[q]["chi2"] - D.sens["sigma_f"][fit][label][q]["chi2"]) < 1e-9
        out[fit]["Z memory (computed, not decided; prespecified sigma_f)"] = {
            q: summarize(D, [z_of(t[q]) for t in D.tests[fit]]) for q in TESTS[3:]
        }
    for q in TESTS:
        blk = (
            out["wls"]["prespecified"]
            if q in X_TESTS
            else out["wls"]["Z memory (computed, not decided; prespecified sigma_f)"]
        )
        assert abs(blk[q]["chi2"] - D.combined["model"][q]["chi2"]) < 1e-9
    return out


# ---------------------------------------------------------------------------------- pool
def pool_of(O, sO, M, verdict: bool = True) -> dict:
    """Eq. (pool): weights 1 / sigma_O^2, error widened by sqrt(chi2_0 / nu) when above one."""
    O, sO, M = (np.asarray(x, float) for x in (O, sO, M))
    lam, se, spread = FT.pooled(O, sO)
    w = 1 / sO**2
    nu = len(O) - 1
    pred = float((w * M).sum() / w.sum())
    out = {
        "Lambda_X": lam,
        "se": se,
        "se_unwidened": float(sqrt(1 / w.sum())),
        "chi2_0_per_dof": spread,
        "dof": nu,
        "birge_ratio": sqrt(max(spread, 1.0)),
        "predicted (same weights)": pred,
        "observed_minus_predicted": lam - pred,
        "observed_over_predicted": lam / pred,
    }
    if verdict:
        t = float(stats.t.ppf(0.95, nu))
        out.update(
            {
                "t_0.95": t,
                "upper_bound (Lambda + t se)": lam + t * se,
                "lower_bound (Lambda - t se)": lam - t * se,
                "verdict": FT.decide(lam, se, t),
            }
        )
    return out


def pool(D: Data) -> dict:
    def terms(fit: str, m=None) -> tuple:
        rows = [t["X lambda"] for t in D.tests[fit]]
        rows = rows if m is None else [r for r, k in zip(rows, m) if k]
        return tuple(
            [r[key] for r in rows] for key in ("observed", "observed_uncertainty", "predicted")
        )

    pre = pool_of(*terms("wls"))
    px = D.combined["pooled_x"]
    assert abs(pre["Lambda_X"] - px["lambda"]) < 1e-12
    assert abs(pre["se"] - px["lambda_uncertainty"]) < 1e-12
    assert abs(pre["chi2_0_per_dof"] - px["spread"]) < 1e-12
    assert abs(pre["t_0.95"] - px["critical"]) < 1e-12
    assert abs(pre["predicted (same weights)"] - px["predicted"]) < 1e-12
    assert pre["verdict"] == px["verdict"]
    pre["statement"] = (
        f"{pre['Lambda_X']:.3f} + {pre['t_0.95']:.3f} x {pre['se']:.3f} = "
        f"{pre['upper_bound (Lambda + t se)']:.3f} < 1: {pre['verdict']}"
    )
    lam = [t["X lambda"] for t in D.tests["wls"]]
    upper = [r["observed"] + ONE_SIDED * r["observed_uncertainty"] for r in lam]
    verdicts = [r["verdict"] for r in D.reps]
    return {
        "rule": "Eq. (pool), run.py: weights w_i = 1 / sigma_O,i^2 over the ten pair-days (WL); "
        "error sqrt(max(1, chi2_0 / nu) / sum w), chi2_0 = sum w (Lambda_i - Lambda)^2, nu = 9; "
        "d = 5 worse if Lambda + t_{nu,0.95} se < 1, better if Lambda - t se > 1. The pooled "
        "prediction uses the same weights.",
        "prespecified": pre,
        "per_pair_day_verdicts (Lambda_X + 1.645 sigma_O < 1)": {
            v: verdicts.count(v) for v in dict.fromkeys(verdicts)
        },
        "largest Lambda_X + 1.645 sigma_O": max(upper),
        "after the run": {
            **{
                f"pair {p} (WL, same rule)": pool_of(*terms("wls", D.mask(p)), verdict=False)
                for p in PAIRS
            },
            "binomial fit, all ten (same rule)": pool_of(*terms("binomial")),
        },
    }


# ------------------------------------------------------------------------------- sigma_f
def idle_r(x, y, nx, ny) -> np.ndarray:
    """sqrt(<X>^2 + <Y>^2) with the shot noise taken out, as run._idle_dephasing."""
    r2 = x**2 + y**2 - (1 - x**2) / nx - (1 - y**2) / ny
    return np.sqrt(np.maximum(r2, 0))


def residual_chi2(seen, log_x0, log_x1, shots, share: float) -> tuple[float, int]:
    """The weighted residual chi2 of run._fit_share's line at its fitted share, and its dof.

    The line is log seen = a + log_x0 - f (log_x0 - log_x1) over the idle lengths with seen >
    0.05, weights 1 / var(log seen); with one length a = 0. Degrees of freedom: lengths - 2.
    """
    keep = seen > 0.05
    extra = (log_x0 - np.log(np.where(keep, seen, 1)))[keep]
    full = (log_x0 - log_x1)[keep]
    weight = (shots * seen**2 / (1 - np.minimum(seen, 0.999) ** 2))[keep]
    m = int(keep.sum())
    if m < 3:
        return 0.0, 0
    a = ((weight * extra).sum() - share * (weight * full).sum()) / weight.sum()
    return float((weight * (extra - a - share * full) ** 2).sum()), m - 2


def idle_fit_scatter() -> dict:
    """Do the per-qubit idle fits scatter beyond their shot noise, and what does that do to the
    uniformity statistic? Reads results/idle_f_inputs.json and each job's analysis.json."""
    inputs = result("idle_f_inputs.json")
    chi, dof = [], []
    per_job = {}
    for name, job in inputs.items():
        idle = read(RUNS / name / "analysis.json")["idle_test"]
        shares, sigmas, red = [], [], []
        for key, v in job["qubits"].items():
            x, y, nx, ny, l0, l1 = (
                np.array(v[k], float) for k in ("x", "y", "nx", "ny", "log_x0", "log_x1")
            )
            r = idle_r(x, y, nx, ny)
            shots = (nx + ny) / 2
            share, sigma = R._fit_share(r, l0, l1, shots)
            d, q = key.split("/")
            stored = idle["per_qubit"][d][int(q)]
            if not (np.isfinite(share) and np.isfinite(sigma)):
                assert stored is None or not np.isfinite(stored)
                continue
            assert abs(share - stored) < 1e-9
            c, k = residual_chi2(r, l0, l1, shots, share)
            shares.append(share)
            sigmas.append(sigma)
            red.append(c / k if k else 1.0)
            if k:
                chi.append(c)
                dof.append(k)
        shares, sigmas, red = (np.array(x) for x in (shares, sigmas, red))
        before = R._uniformity(shares, sigmas)["chi2_per_dof"]
        assert abs(before - idle["uniformity"]["chi2_per_dof"]) < 1e-9
        after = R._uniformity(shares, sigmas * np.sqrt(np.maximum(red, 1.0)))["chi2_per_dof"]
        per_job[name] = {
            "qubits": len(shares),
            "uniformity_chi2_per_dof": before,
            "with_scaled_errors": after,
            "median_reduced_chi2": float(np.median(red)),
        }
    chi, dof = np.array(chi), np.array(dof)
    red = chi / dof
    p = stats.chi2.sf(chi, dof)
    return {
        "what": "each qubit-job's line in run._fit_share (Eq. S5): weighted residual chi2 at the "
        "fitted f, dof = idle lengths kept - 2. Uniformity = run._uniformity's chi2/nu about the "
        "inverse-variance mean f, then with each qubit's sigma times sqrt(max(1, chi2/dof)), "
        "qubits with no dof unscaled. The prespecified tests use sigma_f = 1.25 s_f / sqrt(n), "
        "which does not involve these per-qubit sigmas.",
        "fits_with_dof": len(dof),
        "fits_without_dof": int(sum(v["qubits"] for v in per_job.values()) - len(dof)),
        "dof_counts": {str(k): int((dof == k).sum()) for k in sorted(set(dof.tolist()))},
        "median_reduced_chi2": float(np.median(red)),
        "share_p_below_0.05": float((p < 0.05).mean()),
        "share_p_below_0.001": float((p < 0.001).mean()),
        "uniformity_chi2_per_dof_range": [
            min(v["uniformity_chi2_per_dof"] for v in per_job.values()),
            max(v["uniformity_chi2_per_dof"] for v in per_job.values()),
        ],
        "with_scaled_errors_range": [
            min(v["with_scaled_errors"] for v in per_job.values()),
            max(v["with_scaled_errors"] for v in per_job.values()),
        ],
        "per_job": per_job,
    }


def sigma_f(D: Data) -> dict:
    keep = ("qubits", "f", "sigma_f_prespecified", "sigma_f_idle_noise", "ratio")
    per_job = {name: {k: v[k] for k in keep} for name, v in D.jobs.items()}

    def span(k: str) -> list:
        return [min(v[k] for v in per_job.values()), max(v[k] for v in per_job.values())]

    return {
        "what": "prespecified sigma_f = 1.25 s_f / sqrt(n), the standard error of a median from "
        "the spread of the job's n per-qubit f; idle_noise sigma_f = sd of the median over 2000 "
        "parametric redraws of the idle test (sensitivity.py); ratio = idle_noise / prespecified",
        "prespecified_range": span("sigma_f_prespecified"),
        "idle_noise_range": span("sigma_f_idle_noise"),
        "ratio_range": span("ratio"),
        "per_job": per_job,
        "per_qubit_fits_overdispersed": idle_fit_scatter(),
    }


# ------------------------------------------------------------------------------- windows
def windows(D: Data) -> dict:
    w = D.sens["windows"]
    cells = w["by_choice"]
    kept = [k for k in cells if not k.endswith("3-8")]
    out: dict = {
        "what": "plain means over each pair's five days of Lambda_X, device and model (the "
        "prediction at the measured f), both curves fitted on the same window by the same fit "
        "(sensitivity.py). Systematic = half the range over the ten choices below; rounds 3-8 "
        "are left out because there the distance-5 P(n) is 0.38-0.51 throughout and Lambda_X "
        "mostly noise.",
        "choices": kept,
    }
    for label in ("pair 1", "pair 2", "all ten"):
        key = label.replace("pair", "placement")
        blk: dict = {
            "by_choice": {k: {x: cells[k][key][x] for x in ("device", "model")} for k in cells}
        }
        for x in ("device", "model", "device_minus_model"):
            some = np.array([cells[k][key][x] for k in kept])
            every = np.array([cells[k][key][x] for k in cells])
            syst = float(np.ptp(some) / 2)
            assert abs(syst - w["shifts"][key][x]["systematic (half range without 3-8)"]) < 1e-12
            blk[x] = {
                "systematic (half range, rounds 3-8 left out)": syst,
                "range": [float(some.min()), float(some.max())],
                "half range with rounds 3-8": float(np.ptp(every) / 2),
            }
        out[label] = blk
    out["leaving out round 1 (rounds 2-8 minus 1-8)"] = {
        fit: {
            k.replace("placement", "pair"): v
            for k, v in w["leaving_out_round_1 (2-8 minus 1-8)"][fit].items()
        }
        for fit in FITS
    }
    return out


# ---------------------------------------------------------------------------- table pairs
def shown(v: float, s: float, q: str) -> str:
    return f"{v:.2f} +- {s:.2f}" if q == "lambda" else f"{100 * v:.1f} +- {100 * s:.1f}"


def table_pairs(D: Data, win: dict) -> dict:
    """The pairs' table with the widening applied to the predictions too (effect_sizes.table
    widens the observations only)."""
    es = result("effect_sizes.json")["table"]["cells"]
    noise = {fit: x_rows(D, fit, "idle_noise_only") for fit in FITS}
    out: dict = {
        "convention": "plain means over the five days. Error sqrt(sum sigma_i^2) / 5, times "
        "sqrt(chi2_0 / 4) when chi2_0 / 4 > 1, chi2_0 = sum ((v_i - mean) / sigma_i)^2, for "
        "observations and predictions alike. Observed sigma_i: each fit's own (WL delta method in "
        "X memory and the WL bootstrap sd in Z memory, binomial profile), as effect_sizes.py. "
        "Predicted sigma_i: the test's sigma_M with the prespecified sigma_f (and, in X memory, "
        "with the noise-only sigma_f as a check). For pair 1's Lambda_X the widened prediction "
        "error is the calibration's change from day to day, not the uncertainty of one prediction.",
        "cells": {},
    }
    for b in "XZ":
        for p in PAIRS:
            idx = np.flatnonzero(D.mask(p))
            keys = [D.keys[i] for i in idx]
            for fit in FITS:
                for q in ("eps3", "eps5", "lambda"):
                    test = f"{b} {q}"
                    xs = [D.per_rep[k][b][q] for k in keys]
                    v = [ES.value(x, fit) for x in xs]
                    if b == "Z" and fit == "wls":  # the bootstrap sd, as effect_sizes.table
                        sig = [
                            float(x["bootstrap_wls_sd"])
                            if q == "lambda"
                            else float(D.curves[f"{k} {b} d{q[-1]}"]["bootstrap"]["wls_sd"])
                            for k, x in zip(keys, xs)
                        ]
                    else:
                        sig = [ES.own_sigma(x, fit, q) for x in xs]
                    obs = ES.plain_mean(v, sig)
                    ref = es[f"{b} placement {p} {fit} {q}"]
                    assert abs(obs["value"] - ref["observed"]["value"]) < 1e-12
                    assert abs(obs["se"] - ref["observed"]["se"]) < 1e-12
                    rows = [D.tests[fit][i][test] for i in idx]
                    M = [r["predicted"] for r in rows]
                    assert abs(np.mean(M) - ref["predicted"]["mean"]) < 1e-9
                    pred = ES.plain_mean(M, [sigma_M(r) for r in rows])
                    cell: dict = {
                        "observed": {**obs, "shown": shown(obs["value"], obs["se"], q)},
                        "predicted": {
                            **pred,
                            "range": [float(min(M)), float(max(M))],
                            "shown": shown(pred["value"], pred["se"], q),
                        },
                    }
                    if b == "X":
                        alt = ES.plain_mean(M, [sigma_M(noise[fit][i][test]) for i in idx])
                        cell["predicted"]["with the noise-only sigma_f"] = {
                            k: alt[k] for k in ("se", "se_unwidened", "chi2_per_dof")
                        }
                    if test == "X lambda":
                        sy = win[f"pair {p}"]
                        dev = sy["device"]["systematic (half range, rounds 3-8 left out)"]
                        mod = sy["model"]["systematic (half range, rounds 3-8 left out)"]
                        cell["observed"]["systematic (fit window)"] = dev
                        cell["predicted"]["systematic (fit window)"] = mod
                        cell["observed"]["shown"] += f" (stat) +- {dev:.2f} (syst)"
                    out["cells"][f"{b} pair {p} {fit} {q}"] = cell
    return out


# --------------------------------------------------------------------------------- power
def chi2_power(nc: float, dof: int) -> float:
    return float(stats.ncx2.sf(stats.chi2.ppf(0.95, dof), dof, nc))


def one_power(delta: float, sigma) -> np.ndarray:
    """P(|z| >= 2) for one pair-day whose O - M is off by delta, at its sigma."""
    s = np.asarray(delta / np.asarray(sigma, float))
    return stats.norm.sf(2 - s) + stats.norm.cdf(-2 - s)


def power(D: Data) -> dict:
    nc80 = brentq(lambda x: chi2_power(x, 10) - 0.8, 1e-3, 100.0)
    out: dict = {
        "what": f"the same miss in every pair-day (WL, the prespecified fit); sigma_i = "
        f"sqrt(sigma_O^2 + sigma_M^2). One pair-day: P(|z| >= 2) at a Lambda_X miss of {MISS}. "
        "A pair: noncentral chi2, 5 dof, bound 11.07. All ten: noncentral chi2, 10 dof, bound "
        "18.31, noncentrality sum (miss / sigma_i)^2; and the miss found with 80% probability.",
    }
    for label in SIGMA_F:
        rows = x_rows(D, "wls", label)
        sig = {
            q: np.array(
                [sqrt(r[q]["observed_uncertainty"] ** 2 + sigma_M(r[q]) ** 2) for r in rows]
            )
            for q in X_TESTS
        }
        M = {q: np.array([r[q]["predicted"] for r in rows]) for q in X_TESTS}
        s = sig["X lambda"]
        one = one_power(MISS, s)
        inv = float((1 / s**2).sum())
        blk: dict = {
            "Lambda_X": {
                "per_pair_day": [
                    {
                        "day": int(D.day[i]),
                        "pair": int(D.pair[i]),
                        "sigma": float(s[i]),
                        "power_one_pair_day": float(one[i]),
                    }
                    for i in range(len(s))
                ],
                **{
                    f"pair {p}": {
                        "median_sigma": float(np.median(s[D.mask(p)])),
                        "power_one_pair_day_range": [
                            float(one[D.mask(p)].min()),
                            float(one[D.mask(p)].max()),
                        ],
                        "power_chi2_5dof": chi2_power(MISS**2 * (1 / s[D.mask(p)] ** 2).sum(), 5),
                    }
                    for p in PAIRS
                },
                "median_sigma": float(np.median(s)),
                "power_chi2_10dof": chi2_power(MISS**2 * inv, 10),
                "miss_found_at_80pct_chi2_10dof": sqrt(nc80 / inv),
            },
            **{
                q: {
                    "relative_miss_found_at_80pct_chi2_10dof": sqrt(
                        nc80 / float(((M[q] / sig[q]) ** 2).sum())
                    )
                }
                for q in ("X eps3", "X eps5")
            },
        }
        assert (
            abs(
                blk["Lambda_X"]["power_chi2_10dof"]
                - D.sens["power"][label]["X lambda"]["absolute"][f"{MISS:.2f}"]
            )
            < 1e-9
        )
        out[label] = blk
    design = result("power.json")["f"]
    out[
        "design study (docs/power.py; f known, sigma_O at the predicted curves; run after the run)"
    ] = {
        f"f = {f}": {
            reps: {
                "sigma_one_pair_day": v[reps]["sigma_one_rep"],
                "Lambda_miss": v[reps]["gap_caught_80pct"]["gap"],
                "caught": v[reps]["gap_caught_80pct"]["caught"],
            }
            for reps in ("5 reps", "10 reps")
        }
        for f, v in design.items()
    }
    return out


# ------------------------------------------------------------------------------ headline
def headline(res: dict) -> dict:
    c = res["chi2"]["wls"]
    cz = c["Z memory (computed, not decided; prespecified sigma_f)"]
    cb = res["chi2"]["binomial"]
    pl = res["pool"]["prespecified"]
    pw = res["power"]
    od = res["sigma_f"]["per_qubit_fits_overdispersed"]
    pairs = {f"pair {p}": res["windows"][f"pair {p}"] for p in PAIRS}
    drop1 = res["windows"]["leaving out round 1 (rounds 2-8 minus 1-8)"]["wls"]["all ten"]

    def row(blk: dict) -> list:
        return [blk["chi2"], blk["p"], blk["abs_z_below_2"]]

    return {
        "X tests (WL) [chi2, p, |z| < 2] prespecified sigma_f / idle-noise sigma_f / no f term": {
            q: [row(c[label][q]) for label in SIGMA_F] for q in X_TESTS
        },
        "Holm p over the three X tests, prespecified / idle-noise sigma_f": {
            q: [c[label][q]["p_holm_over_the_three_X_tests"] for label in SIGMA_F[:2]]
            for q in X_TESTS
        },
        "Z tests (WL, computed, not decided) [chi2, p]": {
            q: [cz[q]["chi2"], cz[q]["p"]] for q in TESTS[3:]
        },
        "per pair chi2 (5 dof) [chi2, p] prespecified / idle-noise sigma_f": {
            fit: {
                q: {
                    f"pair {p}": [
                        [
                            res["chi2"][fit][label][q]["per_pair"][f"pair {p}"][k]
                            for k in ("chi2", "p")
                        ]
                        for label in SIGMA_F[:2]
                    ]
                    for p in PAIRS
                }
                for q in X_TESTS
            }
            for fit in FITS
        },
        "binomial X tests [chi2, p, |z| < 2] prespecified / idle-noise sigma_f": {
            q: [row(cb[label][q]) for label in SIGMA_F[:2]] for q in X_TESTS
        },
        "pool": {
            "statement": pl["statement"],
            "Lambda_X, se, chi2_0/nu, t_0.95, upper bound, predicted": [
                pl["Lambda_X"],
                pl["se"],
                pl["chi2_0_per_dof"],
                pl["t_0.95"],
                pl["upper_bound (Lambda + t se)"],
                pl["predicted (same weights)"],
            ],
            "per pair-day verdicts": res["pool"][
                "per_pair_day_verdicts (Lambda_X + 1.645 sigma_O < 1)"
            ],
            "within pair chi2_0/nu [pair 1, pair 2]": [
                res["pool"]["after the run"][f"pair {p} (WL, same rule)"]["chi2_0_per_dof"]
                for p in PAIRS
            ],
        },
        "sigma_f [prespecified range, idle-noise range]": [
            res["sigma_f"]["prespecified_range"],
            res["sigma_f"]["idle_noise_range"],
        ],
        "per-qubit idle fits: median reduced chi2, share p < 0.05, uniformity range, scaled": [
            od["median_reduced_chi2"],
            od["share_p_below_0.05"],
            od["uniformity_chi2_per_dof_range"],
            od["with_scaled_errors_range"],
        ],
        "fit-window systematic on Lambda_X [device, model; device half range with 3-8]": {
            k: [
                v["device"]["systematic (half range, rounds 3-8 left out)"],
                v["model"]["systematic (half range, rounds 3-8 left out)"],
                v["device"]["half range with rounds 3-8"],
            ]
            for k, v in pairs.items()
        },
        "leaving out round 1, all ten (WL) [device, se, model]": [
            drop1[k] for k in ("device", "device_se_over_repetitions", "model")
        ],
        "table pairs (shown)": {
            k: {"observed": v["observed"]["shown"], "predicted": v["predicted"]["shown"]}
            for k, v in res["table_pairs"]["cells"].items()
        },
        f"power at a Lambda_X miss of {MISS} [one pair-day range pair 1, pair 2; chi2 10 dof]": {
            label: [
                pw[label]["Lambda_X"]["pair 1"]["power_one_pair_day_range"],
                pw[label]["Lambda_X"]["pair 2"]["power_one_pair_day_range"],
                pw[label]["Lambda_X"]["power_chi2_10dof"],
            ]
            for label in SIGMA_F
        },
    }


def main() -> None:
    D = Data()
    win = windows(D)
    res = {
        "per_pair_day": per_pair_day(D),
        "chi2": chi2_tests(D),
        "pool": pool(D),
        "sigma_f": sigma_f(D),
        "windows": win,
        "table_pairs": table_pairs(D, win),
        "power": power(D),
    }
    res = {"headline": headline(res), **res}
    h = res["headline"]
    print(h["pool"]["statement"])
    for q, v in h[
        "X tests (WL) [chi2, p, |z| < 2] prespecified sigma_f / idle-noise sigma_f / no f term"
    ].items():
        print(f"{q:9s} " + "  ".join(f"{x:6.1f} p {p:.2g} ({n})" for x, p, n in v))
    for k, v in h["table pairs (shown)"].items():
        print(f"{k:28s} {v['observed']:34s} pred {v['predicted']}")
    write_json("prespecified_tests.json", res)


if __name__ == "__main__":
    main()
