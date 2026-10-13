"""How far the model is from the device, and what the agreement test could see, from stored
results only (Sec. agreement, the model-versus-device table; Table placement; App. stats;
Fig. A_effect_sizes).

The prespecified test, chi2 = sum_i z_i^2 with z = (O - M) / sqrt(sigma_O^2 + sigma_M^2), answers
pass or fail. This script measures the misses and what the test could have seen. O/M is observed
over predicted; a repetition is one day and placement (repetition 1 is placement 1).

  ratios     O/M per placement: geometric mean over the five days with a Student t (4 dof) 95%
             interval; over the ten repetitions: DerSimonian-Laird random-effects mean of ln(O/M),
             variances (sigma_O/O)^2 + (sigma_M/M)^2, Hartung-Knapp 95% interval, and tau, the
             spread of ln(O/M) between repetitions beyond their errors. Both fits; binomial primary
  chi2       the prespecified chi2 with p-values (Holm over the three X tests); chi2 = 10 zbar^2
             + sum (z - zbar)^2, bias (1 dof) and scatter (9 dof); per placement (5 dof); the same
             with the f term of sigma_M set to zero, and how many repetitions agree (|z| < 2)
  budget     sigma_M^2 = sigma_sim^2 + (sigma_f dM/df)^2 + Delta_cal^2 / 2 term by term beside
             sigma_O; a split repetition adds its two jobs' relative terms in quadrature
             (run._ratio_test), a paired one takes Lambda's own slope in f
  power      each X test's power at the bounds actually used (noncentral chi2, 10 dof) against a
             uniform discrepancy, and the discrepancy it finds with 80% probability, beside the
             idealized power.json (f known exactly, sigma_O at the predicted curves)
  tracking   does the device follow the day-to-day changes of the prediction? chi2 about a
             constant, sigma_O only, against O_i = c M_i with sigma_i^2 = sigma_O,i^2 + c^2
             sigma_M,i^2 (4 dof each); Pearson r with a Fisher interval; each day left out
  prejob     the observations against the predictions stored before each job at f = 0 and f = 1
  table      Table placement under one convention: every cell the plain mean over the five days,
             error sqrt(sum sigma_i^2) / 5, times sqrt(chi2/nu) when chi2/nu > 1 (nu = 4), sigma_i
             each estimator's own (the WL bootstrap sd in Z memory); the inverse-variance means
             beside it
  errors     the weighted-log (WL) delta-method sigma against the parametric bootstrap stored in
             fits.json and the binomial profile, per curve, and the rounds each eps used
  window     Lambda_X from rounds 2-8 against 1-8, and the prespecified test refitted from the
             stored counts on other windows of rounds
  selection  the excess per round in percentage points at d = 3 and d = 5, and the fixed
             (placement 1) against the re-selected (placement 2) d = 3 patch
  pool       the prespecified ten-repetition pool and its prediction (App. stats record)

Reads runs/combined.json, results/fits.json, results/fits_counts.json, results/power.json and the
jobs' analysis.json; no simulation; a few seconds in one process.

    python docs/analysis/effect_sizes.py      # writes results/effect_sizes.json
"""

from __future__ import annotations

from math import atanh, exp, sqrt, tanh
from types import SimpleNamespace

import numpy as np
from scipy import stats
from scipy.optimize import brentq, minimize_scalar

import fits as FT
import fitting as F
from common import RUNS, day_number, read, result, write_json

TESTS = ("X eps3", "X eps5", "X lambda", "Z eps3", "Z eps5", "Z lambda")
X_TESTS = TESTS[:3]
FITS = ("binomial", "wls")  # the binomial fit is unbiased; it is primary for effect sizes
PLACEMENTS = (1, 2)
PRED = "predicted_at_measured_dephasing"
AFTER = "predicted_from_calibration_after"
FROZEN = ("frozen_f0.0", "frozen_f1.0")
WINDOWS = {"1-8": (1, 8), "2-8": (2, 8), "3-8": (3, 8), "1-6": (1, 6), "2-6": (2, 6), "1-4": (1, 4)}
CRIT = float(stats.chi2.ppf(0.95, 10))


def floats(x) -> list:
    return [float(v) for v in x]


# ----------------------------------------------------------------------------------- data
class Data:
    def __init__(self) -> None:
        self.combined = read(RUNS / "combined.json")
        self.fits = result("fits.json")
        self.counts = result("fits_counts.json")
        self.power = result("power.json")
        reps = self.combined["reps"]  # day, then repetition
        self.keys = [f"{r['day']} r{r['rep']}" for r in reps]
        self.day = np.array([day_number(r["day"]) for r in reps])
        self.placement = np.array([r["rep"] for r in reps])
        self.pairing = [r["pairing"] for r in reps]
        self.tests = {
            "wls": [r["tests"] for r in reps],  # the prespecified analysis itself
            "binomial": [self.fits["agreement_tests"]["binomial"][k] for k in self.keys],
        }
        for k, t in zip(self.keys, self.tests["wls"]):
            again = self.fits["agreement_tests"]["wls"][k]
            assert all(
                abs(t[q][x] - again[q][x]) < 1e-9
                for q in TESTS
                for x in ("observed", "predicted", "bound")
            )

    def terms(self, fit: str, q: str) -> SimpleNamespace:
        """O, M and the bound's terms over the ten repetitions."""
        rows = [t[q] for t in self.tests[fit]]

        def get(key: str) -> np.ndarray:
            return np.array([r[key] for r in rows], float)

        s = SimpleNamespace(
            O=get("observed"),
            M=get("predicted"),
            sO=get("observed_uncertainty"),
            sim=get("simulator_noise"),
            ff=np.abs(get("from_f")),
            gap=get("calibration_gap"),
            half=get("bound") / 2,
        )
        s.sM = np.sqrt(s.sim**2 + s.ff**2 + s.gap**2 / 2)
        assert np.allclose(s.half, np.hypot(s.sO, s.sM), rtol=1e-12)
        return s

    def mask(self, placement: int) -> np.ndarray:
        return self.placement == placement


# ---------------------------------------------------------------------------- statistics
def t_interval(y) -> dict:
    """exp(mean y) with the Student t 95% interval from the spread of y."""
    y = np.asarray(y, float)
    n = len(y)
    m, se = float(y.mean()), float(y.std(ddof=1) / sqrt(n))
    t = float(stats.t.ppf(0.975, n - 1))
    return {"ratio": exp(m), "ci95": [exp(m - t * se), exp(m + t * se)], "n": n}


def random_effects(y, v) -> dict:
    """DerSimonian-Laird mean of y with sampling variances v, exponentiated, with the
    Hartung-Knapp-Sidik-Jonkman 95% interval (t, k - 1 dof); also that interval with its
    variance factor not allowed below one, the cautious variant."""
    y, v = np.asarray(y, float), np.asarray(v, float)
    k, w = len(y), 1 / v
    fixed = (w * y).sum() / w.sum()
    Q = float((w * (y - fixed) ** 2).sum())
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w**2).sum() / w.sum()))
    ws = 1 / (v + tau2)
    mu = float((ws * y).sum() / ws.sum())
    q = float((ws * (y - mu) ** 2).sum() / (k - 1))
    t = float(stats.t.ppf(0.975, k - 1))
    hk, floor = t * sqrt(q / ws.sum()), t * sqrt(max(q, 1.0) / ws.sum())
    return {
        "ratio": exp(mu),
        "ci95": [exp(mu - hk), exp(mu + hk)],
        "ci95_factor_at_least_1": [exp(mu - floor), exp(mu + floor)],
        "tau": sqrt(tau2),
        "I2": max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0,
        "Q": Q,
        "k": k,
    }


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    out, running = [0.0] * len(p), 0.0
    for i, j in enumerate(order):
        running = max(running, min(1.0, (len(p) - i) * p[j]))
        out[j] = running
    return out


def chi2_p(x: float, dof: int) -> dict:
    return {"chi2": float(x), "dof": dof, "p": float(stats.chi2.sf(x, dof))}


def pearson(o, m) -> dict:
    r = float(np.corrcoef(o, m)[0, 1])
    se = 1 / sqrt(len(o) - 3)
    return {"r": r, "ci95": [tanh(atanh(r) - 1.96 * se), tanh(atanh(r) + 1.96 * se)], "n": len(o)}


# -------------------------------------------------------------------------------- ratios
def ratios(D: Data) -> dict:
    out: dict = {}
    for fit in FITS:
        out[fit] = {}
        for q in TESTS:
            s = D.terms(fit, q)
            y = np.log(s.O / s.M)
            v = (s.sO / s.O) ** 2 + (s.sM / s.M) ** 2
            out[fit][q] = {
                **{f"placement {p}": t_interval(y[D.mask(p)]) for p in PLACEMENTS},
                "all ten": random_effects(y, v),
                "all ten, t on the ten ratios": t_interval(y),
                "per_repetition": floats(np.exp(y)),
            }
    return out


# ---------------------------------------------------------------------------------- chi2
def chi2_tests(D: Data) -> dict:
    out: dict = {}
    for fit in FITS:
        res = {}
        for q in TESTS:
            s = D.terms(fit, q)
            z = (s.O - s.M) / s.half
            z0 = (s.O - s.M) / np.sqrt(s.sO**2 + s.sim**2 + s.gap**2 / 2)
            zbar = float(z.mean())

            def split(zz):
                return {
                    f"placement {p}": chi2_p((zz[D.mask(p)] ** 2).sum(), int(D.mask(p).sum()))
                    for p in PLACEMENTS
                }

            res[q] = {
                **chi2_p((z**2).sum(), len(z)),
                "z": floats(z),
                "agree": int((np.abs(z) < 2).sum()),
                "z_positive": int((z > 0).sum()),
                "mean_z": zbar,
                "bias": chi2_p(len(z) * zbar**2, 1),
                "scatter": chi2_p(((z - zbar) ** 2).sum(), len(z) - 1),
                "per_placement": split(z),
                "without_f_term": {
                    **chi2_p((z0**2).sum(), len(z0)),
                    "z": floats(z0),
                    "agree": int((np.abs(z0) < 2).sum()),
                    "per_placement": split(z0),
                },
            }
        for q, p in zip(X_TESTS, holm([res[q]["p"] for q in X_TESTS])):
            res[q]["p_holm_over_the_three_X_tests"] = p
        out[fit] = res
    for q in TESTS:
        assert abs(out["wls"][q]["chi2"] - D.combined["model"][q]["chi2"]) < 1e-9
    return out


# -------------------------------------------------------------------------------- budget
def budget(D: Data) -> dict:
    out: dict = {
        "what": "sigma_M^2 = sigma_sim^2 + from_f^2 + gap^2 / 2; bound / 2 = sqrt(sigma_O^2 + "
        "sigma_M^2); f_share = from_f^2 / (bound / 2)^2. IBM did not recalibrate during any job, "
        "so gap = 0 throughout."
    }
    for fit in FITS:
        out[fit] = {}
        for q in TESTS:
            s = D.terms(fit, q)
            share = s.ff**2 / s.half**2
            blk: dict = {
                "per_repetition": [
                    {
                        "day": int(D.day[i]),
                        "placement": int(D.placement[i]),
                        "pairing": D.pairing[i],
                        "sigma_O": float(s.sO[i]),
                        "sigma_sim": float(s.sim[i]),
                        "from_f": float(s.ff[i]),
                        "gap": float(s.gap[i]),
                        "sigma_M": float(s.sM[i]),
                        "bound_half": float(s.half[i]),
                        "f_share": float(share[i]),
                    }
                    for i in range(len(s.O))
                ]
            }
            for p in PLACEMENTS:
                m = D.mask(p)
                blk[f"placement {p}"] = {
                    "median_sigma_O": float(np.median(s.sO[m])),
                    "median_sigma_sim": float(np.median(s.sim[m])),
                    "from_f_range": [float(s.ff[m].min()), float(s.ff[m].max())],
                    "median_sigma_M": float(np.median(s.sM[m])),
                    "median_f_share": float(np.median(share[m])),
                }
            out[fit][q] = blk
    # How Lambda's terms come from the eps terms. A split repetition (placement 2) has two jobs
    # with independent f, so its relative terms add in quadrature; a paired one (placement 1)
    # moves eps3 and eps5 with one f, and Lambda's own slope in f partly cancels them.
    comp = []
    for i, t in enumerate(D.tests["wls"]):
        row: dict = {"day": int(D.day[i]), "placement": int(D.placement[i])}
        for b in "XZ":
            e3, e5, lam = t[f"{b} eps3"], t[f"{b} eps5"], t[f"{b} lambda"]
            as_split = FT.ratio_test(e3, e5)
            if D.pairing[i] == "split" or b == "Z":  # Z Lambda is always run._ratio_test
                assert all(abs(as_split[k] - lam[k]) < 1e-12 for k in ("from_f", "bound"))
            row[f"{b} lambda"] = {
                "from_f": abs(lam["from_f"]),
                "from_f_if_eps3_and_eps5_had_independent_f": as_split["from_f"],
                "sigma_O": lam["observed_uncertainty"],
            }
        comp.append(row)
    out["lambda_from_eps_terms"] = comp
    sigma_f: dict = {}
    for name, j in D.counts["jobs"].items():
        qubits = sum(
            len(v) for v in read(RUNS / name / "analysis.json")["idle_test"]["per_qubit"].values()
        )
        kind = (
            "placement 1 (both patches)" if j["rep"] == 1 else f"placement 2, d = {j['patches'][0]}"
        )
        g = sigma_f.setdefault(kind, {"idle_test_qubits": qubits, "sigma_f": []})
        g["sigma_f"].append(j["f_uncertainty"])
    for g in sigma_f.values():
        g["sigma_f_range"] = [min(g["sigma_f"]), max(g["sigma_f"])]
    out["sigma_f_by_job_kind"] = sigma_f
    return out


# --------------------------------------------------------------------------------- power
def power(D: Data) -> dict:
    def pw(nc: float) -> float:
        return float(stats.ncx2.sf(CRIT, 10, nc))

    nc80 = brentq(lambda x: pw(x) - 0.8, 1e-3, 100.0)
    out: dict = {
        "what": "power of chi2 = sum z^2 > 18.31 when every repetition is off by the same "
        "discrepancy, at each repetition's sigma = bound/2 (noncentral chi2, 10 dof, "
        "noncentrality sum (delta_i/sigma_i)^2); the discrepancy found with 80% probability",
        "noncentrality_for_80pct": nc80,
    }
    for fit in FITS:
        out[fit] = {}
        for q in X_TESTS:
            s = D.terms(fit, q)
            blk = {}
            for label, sig in (
                ("realized", s.half),
                ("without_f_term", np.sqrt(s.sO**2 + s.sim**2 + s.gap**2 / 2)),
            ):
                rel = float(((s.M / sig) ** 2).sum())
                b = {
                    "median_sigma": float(np.median(sig)),
                    **{
                        f"median_sigma placement {p}": float(np.median(sig[D.mask(p)]))
                        for p in PLACEMENTS
                    },
                    "relative": {
                        f"{r:.2f}": pw(r * r * rel) for r in (0.05, 0.10, 0.15, 0.20, 0.25)
                    },
                    "relative_found_at_80pct": sqrt(nc80 / rel),
                }
                if q.endswith("lambda"):
                    inv = float((1 / sig**2).sum())
                    b["absolute"] = {
                        f"{d:.2f}": pw(d * d * inv) for d in (0.03, 0.05, 0.06, 0.08, 0.10, 0.12)
                    }
                    b["absolute_found_at_80pct"] = sqrt(nc80 / inv)
                blk[label] = b
            out[fit][q] = blk
    out["idealized_power_json"] = {
        f"f = {f}": {
            reps: {
                "sigma_one_rep": v["sigma_one_rep"],
                "gap": v["gap_caught_80pct"]["gap"],
                "caught": v["gap_caught_80pct"]["caught"],
            }
            for reps, v in blk.items()
            if reps.endswith("reps")
        }
        for f, blk in D.power["f"].items()
    }
    return out


# ------------------------------------------------------------------------------ tracking
def track(O, M, sO, sM) -> dict:
    """Constant device (sigma_O) against a device that follows the prediction, O = c M."""
    w = 1 / sO**2
    mu = float((w * O).sum() / w.sum())
    chi_c = float((w * (O - mu) ** 2).sum())

    def chi_t(c: float) -> float:
        return float((((O - c * M) ** 2) / (sO**2 + c * c * sM**2)).sum())

    c = float(
        minimize_scalar(chi_t, bounds=(0.05, 5.0), method="bounded", options={"xatol": 1e-10}).x
    )
    dof = len(O) - 1
    return {
        "constant": {"value": mu, **chi2_p(chi_c, dof)},
        "follows_prediction": {"c": c, **chi2_p(chi_t(c), dof)},
        "pearson": pearson(O, M),
        "observed_range": [float(O.min()), float(O.max())],
        "predicted_range": [float(M.min()), float(M.max())],
        "relative_sd_observed": float(O.std(ddof=1) / O.mean()),
        "relative_sd_predicted": float(M.std(ddof=1) / M.mean()),
        "predicted_sd_over_rms_sigma_M": float(M.std(ddof=1) / sqrt((sM**2).mean())),
    }


def tracking(D: Data) -> dict:
    out: dict = {
        "what": "placement 1, Lambda_X: its prediction barely depends on f, so the predicted "
        "day-to-day swing comes from the calibration alone"
    }
    for fit in FITS:
        s = D.terms(fit, "X lambda")
        m = D.mask(1)
        O, M, sO, sM, ff = s.O[m], s.M[m], s.sO[m], s.sM[m], s.ff[m]
        blk = track(O, M, sO, sM)
        blk["max_from_f"] = float(ff.max())
        blk["days"] = [int(d) for d in D.day[m]]
        blk["observed"], blk["predicted"] = floats(O), floats(M)
        blk["leave_one_day_out"] = {}
        for i, d in enumerate(D.day[m]):
            k = np.arange(len(O)) != i
            t = track(O[k], M[k], sO[k], sM[k])
            blk["leave_one_day_out"][f"without day {d}"] = {
                "constant": t["constant"],
                "follows_prediction": t["follows_prediction"],
                "pearson_r": t["pearson"]["r"],
            }
        others = {}
        for q in TESTS:
            sq = D.terms(fit, q)
            for p in PLACEMENTS:
                mm = D.mask(p)
                t = track(sq.O[mm], sq.M[mm], sq.sO[mm], sq.sM[mm])
                others[f"{q}, placement {p}"] = {
                    "constant_chi2": t["constant"]["chi2"],
                    "follows_prediction_chi2": t["follows_prediction"]["chi2"],
                    "c": t["follows_prediction"]["c"],
                    "pearson_r": t["pearson"]["r"],
                    "predicted_sd_over_rms_sigma_M": t["predicted_sd_over_rms_sigma_M"],
                }
        blk[
            "every_quantity (4 dof each; informative only where the predicted sd exceeds sigma_M)"
        ] = others
        out[fit] = blk
    return out


# -------------------------------------------------------------------------------- prejob
def prejob(D: Data) -> dict:
    pr, per_rep = D.fits["predictions"], D.fits["per_rep"]
    jobs = D.counts["jobs"]

    def frozen(job: str, src: str, b: str, d: str, fit: str) -> tuple[float, float]:
        c = pr[job][src][b][d][fit]
        return c["eps"], (c["se"] if fit == "wls" else c["se_quasi"])

    out: dict = {
        "what": "predictions stored before each job (prediction.json) at f = 0 and f = 1, fitted "
        "like the data. The observed eps are decoded with priors at the job's measured f.",
        "measured_f": {name: j["f"] for name, j in jobs.items()},
        "jobs_with_measured_f_above_1": sum(j["f"] > 1 for j in jobs.values()),
    }
    for fit in FITS:
        rows, blk = [], {}
        ends: dict = {f: {q: [] for q in TESTS} for f in FROZEN}
        sims: dict = {f: {q: [] for q in TESTS} for f in FROZEN}
        for i, k in enumerate(D.keys):
            j3, j5 = per_rep[k]["jobs"]["3"], per_rep[k]["jobs"]["5"]
            row: dict = {"day": int(D.day[i]), "placement": int(D.placement[i])}
            for src in FROZEN:
                for b in "XZ":
                    e3, s3 = frozen(j3, src, b, "3", fit)
                    e5, s5 = frozen(j5, src, b, "5", fit)
                    lam = e3 / e5
                    vals = {
                        "eps3": (e3, s3),
                        "eps5": (e5, s5),
                        "lambda": (lam, lam * sqrt((s3 / e3) ** 2 + (s5 / e5) ** 2)),
                    }
                    for name, (v, s) in vals.items():
                        ends[src][f"{b} {name}"].append(v)
                        sims[src][f"{b} {name}"].append(s)
                        row[f"{b} {name} {src}"] = v
            for q in TESTS:
                row[f"{q} observed"] = D.tests[fit][i][q]["observed"]
            rows.append(row)
        at0 = {q: np.array(ends[FROZEN[0]][q]) for q in TESTS}
        at1 = {q: np.array(ends[FROZEN[1]][q]) for q in TESTS}
        sim0 = {q: np.array(sims[FROZEN[0]][q]) for q in TESTS}
        sim1 = {q: np.array(sims[FROZEN[1]][q]) for q in TESTS}
        for q in ("X eps3", "X eps5"):
            obs = D.terms(fit, q).O
            blk[q] = {
                "observed_above_f1": int((obs > at1[q]).sum()),
                "observed_above_f0": int((obs > at0[q]).sum()),
                "observed_above_both": int((obs > np.maximum(at0[q], at1[q])).sum()),
                "of": len(obs),
                "O_over_M_f1": {
                    **{
                        f"placement {p}": t_interval(np.log(obs / at1[q])[D.mask(p)])
                        for p in PLACEMENTS
                    },
                    "all ten": t_interval(np.log(obs / at1[q])),
                },
                "O_over_M_f0_all_ten": t_interval(np.log(obs / at0[q])),
            }
        s = D.terms(fit, "X lambda")
        obs, q = s.O, "X lambda"
        lo, hi = np.minimum(at0[q], at1[q]), np.maximum(at0[q], at1[q])
        near = np.clip(obs, lo, hi)  # the end of the bracket nearer the observation
        near_sim = np.where(near == at0[q], sim0[q], sim1[q])
        z = (obs - near) / np.sqrt(s.sO**2 + near_sim**2)
        blk["X lambda"] = {
            "bracket_per_repetition": [[float(a), float(b)] for a, b in zip(lo, hi)],
            "bracket_width": floats(hi - lo),
            "bracket_width_range": {
                f"placement {p}": [
                    float((hi - lo)[D.mask(p)].min()),
                    float((hi - lo)[D.mask(p)].max()),
                ]
                for p in PLACEMENTS
            },
            "observed_above": int((obs > hi).sum()),
            "observed_below": int((obs < lo).sum()),
            "observed_inside": int(((obs >= lo) & (obs <= hi)).sum()),
            "z_against_nearer_end (sigma_O and the prediction's shot noise only)": floats(z),
        }
        zq = {}
        for q in ("Z eps3", "Z eps5", "Z lambda"):
            s = D.terms(fit, q)
            y = np.log(s.O / at1[q])
            v = (s.sO / s.O) ** 2 + (sim1[q] / at1[q]) ** 2
            zq[q] = {
                "max_abs_f1_minus_f0": float(np.abs(at1[q] - at0[q]).max()),
                "O_over_M_f1": {
                    **{f"placement {p}": t_interval(y[D.mask(p)]) for p in PLACEMENTS},
                    "all ten": random_effects(y, v),
                },
            }
        blk["Z memory (does not depend on f)"] = zq
        blk["per_repetition"] = rows
        out[fit] = blk
    return out


# --------------------------------------------------------------------------------- table
def value(x: dict, fit: str) -> float:
    return float(x["wls"][0]) if fit == "wls" else float(x["binomial"])


def profile_sigma(x: dict, q: str) -> float:
    if q == "lambda":
        return float(x["binomial_se_quasi"])
    return float((x["binomial_hi68_quasi"] - x["binomial_lo68_quasi"]) / 2)


def own_sigma(x: dict, fit: str, q: str) -> float:
    return float(x["wls"][1]) if fit == "wls" else profile_sigma(x, q)


def plain_mean(v, s) -> dict:
    v, s = np.asarray(v, float), np.asarray(s, float)
    m = float(v.mean())
    se0 = float(sqrt((s**2).sum()) / len(v))
    chi = float((((v - m) / s) ** 2).sum())
    red = chi / (len(v) - 1)
    return {
        "value": m,
        "se": se0 * sqrt(max(red, 1.0)),
        "se_unwidened": se0,
        "chi2_per_dof": red,
        "widened": red > 1,
    }


def table(D: Data) -> dict:
    per_rep, curves = D.fits["per_rep"], D.fits["curves"]
    out: dict = {
        "convention": "every cell, observed and predicted: plain mean over the five days. Observed "
        "error sqrt(sum sigma_i^2) / 5, multiplied by sqrt(chi2/nu) when chi2/nu > 1, chi2 about "
        "the plain mean, nu = 4. sigma_i is each estimator's own sampling error: binomial, half "
        "the 68% quasi-likelihood profile interval; WL, the delta-method sigma as prespecified in "
        "X memory, and in Z memory, where that sigma is unreliable near P = 1/2 (see errors), the "
        "parametric-bootstrap sd of the WL estimator. Predicted: plain mean of the daily "
        "predictions fitted the same way, with their range and sigma_M = sqrt(sum "
        "sigma_M,i^2) / 5 from the agreement test's sigma_M of each day.",
        "cells": {},
    }
    diffs = []
    for b in "XZ":
        for p in PLACEMENTS:
            keys = [k for k in D.keys if per_rep[k]["placement"] == p]
            for fit in ("wls", "binomial"):
                for q in ("eps3", "eps5", "lambda"):
                    xs = [per_rep[k][b][q] for k in keys]
                    v = [value(x, fit) for x in xs]
                    own = [own_sigma(x, fit, q) for x in xs]
                    boot = [
                        float(x["bootstrap_wls_sd"])
                        if q == "lambda"
                        else float(curves[f"{k} {b} d{q[-1]}"]["bootstrap"]["wls_sd"])
                        for k, x in zip(keys, xs)
                    ]
                    sig = boot if (b == "Z" and fit == "wls") else own
                    pred = np.array([per_rep[k][b]["predicted"][q][fit] for k in keys])
                    sM = D.terms(fit, f"{b} {q}").sM[[D.keys.index(k) for k in keys]]
                    cell = plain_mean(v, sig)
                    iv_v, iv_se, iv_red = FT.pooled(v, own)
                    cell_out = {
                        "observed": cell,
                        "predicted": {
                            "mean": float(pred.mean()),
                            "range": [float(pred.min()), float(pred.max())],
                            "sigma_M": float(sqrt((sM**2).sum()) / len(sM)),
                        },
                        "observed_over_predicted (ratio of the plain means)": cell["value"]
                        / float(pred.mean()),
                        "inverse_variance (each fit's own sigma, Birge-widened)": {
                            "value": iv_v,
                            "se": iv_se,
                            "chi2_per_dof": iv_red,
                        },
                        "iv_minus_plain": iv_v - cell["value"],
                    }
                    if b == "Z" and fit == "wls":
                        cell_out["with the delta-method sigma_i (check)"] = plain_mean(v, own)
                        cell_out["with the binomial profile sigma_i (check)"] = plain_mean(
                            v, [profile_sigma(x, q) for x in xs]
                        )
                    out["cells"][f"{b} placement {p} {fit} {q}"] = cell_out
                    diffs.append(
                        (
                            abs(iv_v - cell["value"]),
                            f"{b} placement {p} {fit} {q}",
                            iv_v,
                            cell["value"],
                        )
                    )
    diffs.sort(reverse=True)
    out["largest_iv_minus_plain"] = [
        {"cell": c, "iv": iv, "plain": pm, "abs_difference": d} for d, c, iv, pm in diffs[:4]
    ]
    out["iv_minus_plain_lambda_X_max"] = max(
        d for d, c, _, _ in diffs if c.startswith("X") and c.endswith("lambda")
    )
    return out


# -------------------------------------------------------------------------------- errors
def errors(D: Data) -> dict:
    curves, per_rep = D.fits["curves"], D.fits["per_rep"]
    rounds = D.counts["rounds"]
    rows: dict = {}
    for name, c in curves.items():
        w, bi, bs = c["wls"], c["binomial"], c["bootstrap"]
        rows[name] = {
            "wls_eps": w["eps"],
            "wls_sigma": w["se"],
            "wls_sigma_shot_noise": w["se_shot_noise"],
            "wls_bootstrap_sd": bs["wls_sd"],
            "wls_bootstrap_sd_over_sigma": bs["wls_sd"] / w["se"],
            "binomial_eps": bi["eps"],
            "binomial_profile_sigma": bi["se_quasi"],
            "binomial_bootstrap_sd": bs["binomial_sd"],
            "binomial_bootstrap_sd_over_profile_sigma": bs["binomial_sd"] / bi["se_quasi"],
            "rounds_used_by_wls": [n for n, P in zip(rounds, c["P"]) if 0 < P < 0.5],
        }
    summary: dict = {}
    for b in "XZ":
        for d in "35":
            rs = [r for n, r in rows.items() if n.endswith(f"{b} d{d}")]
            used = [len(r["rounds_used_by_wls"]) for r in rs]
            summary[f"{b} eps{d}"] = {
                "wls_bootstrap_sd_over_sigma": [
                    float(np.median([r["wls_bootstrap_sd_over_sigma"] for r in rs])),
                    float(min(r["wls_bootstrap_sd_over_sigma"] for r in rs)),
                    float(max(r["wls_bootstrap_sd_over_sigma"] for r in rs)),
                ],
                "binomial_bootstrap_sd_over_profile_sigma": [
                    float(np.median([r["binomial_bootstrap_sd_over_profile_sigma"] for r in rs])),
                    float(min(r["binomial_bootstrap_sd_over_profile_sigma"] for r in rs)),
                    float(max(r["binomial_bootstrap_sd_over_profile_sigma"] for r in rs)),
                ],
                "wls_rounds_used_counts": {str(n): used.count(n) for n in sorted(set(used))},
            }
    lam: dict = {}
    for b in "XZ":
        rs = []
        for k in D.keys:
            x = per_rep[k][b]["lambda"]
            rs.append(
                {
                    "repetition": k,
                    "wls": x["wls"][0],
                    "wls_sigma": x["wls"][1],
                    "wls_bootstrap_sd": x["bootstrap_wls_sd"],
                    "binomial": x["binomial"],
                    "binomial_profile_sigma": x["binomial_se_quasi"],
                    "binomial_bootstrap_sd": x["bootstrap_sd"],
                }
            )
        r1 = np.array([r["wls_bootstrap_sd"] / r["wls_sigma"] for r in rs])
        r2 = np.array([r["binomial_bootstrap_sd"] / r["binomial_profile_sigma"] for r in rs])
        lam[f"Lambda_{b}"] = {
            "per_repetition": rs,
            "wls_bootstrap_sd_over_sigma [median, min, max]": [
                float(np.median(r1)),
                float(r1.min()),
                float(r1.max()),
            ],
            "binomial_bootstrap_sd_over_profile_sigma [median, min, max]": [
                float(np.median(r2)),
                float(r2.min()),
                float(r2.max()),
            ],
        }
    # what the WL Z tests and the Birge widening become with the bootstrap sd as sigma_O
    consequences: dict = {}
    for q, getter in (
        ("Z lambda", lambda k: per_rep[k]["Z"]["lambda"]["bootstrap_wls_sd"]),
        ("Z eps5", lambda k: D.fits["curves"][f"{k} Z d5"]["bootstrap"]["wls_sd"]),
        ("X lambda", lambda k: per_rep[k]["X"]["lambda"]["bootstrap_wls_sd"]),
    ):
        s = D.terms("wls", q)
        boot = np.array([getter(k) for k in D.keys])
        z = (s.O - s.M) / np.sqrt(boot**2 + s.sM**2)
        consequences[q] = {
            "chi2_as_quoted": float((((s.O - s.M) / s.half) ** 2).sum()),
            "chi2_with_bootstrap_sigma_O": float((z**2).sum()),
            "p": float(stats.chi2.sf((z**2).sum(), 10)),
            **{
                f"placement {p} chi2_per_dof about the inverse-variance mean, quoted -> bootstrap": [
                    FT.pooled(s.O[D.mask(p)], s.sO[D.mask(p)])[2],
                    FT.pooled(s.O[D.mask(p)], boot[D.mask(p)])[2],
                ]
                for p in PLACEMENTS
            },
        }
    return {
        "what": "ratios are bootstrap sd / quoted sigma: above 1, the quoted sigma is too small. "
        "Parametric bootstrap, 2,000 replicates per curve drawn from its binomial fit (fits.json).",
        "per_curve_summary": summary,
        "lambda": lam,
        "wls_with_bootstrap_sigma": consequences,
        "per_curve": rows,
    }


# -------------------------------------------------------------------------------- window
def wls_curve(rounds, k, N) -> dict:
    w = F.wls(rounds, np.asarray(k, float) / N, N)
    return {"wls": {"eps": w.eps, "se": w.se}}


def window_test(D: Data, lo: int, hi: int) -> dict:
    """The prespecified test (WL) refitted on rounds lo..hi from the stored counts; the
    binomial point estimates of O/M beside it."""
    rounds = np.array(D.counts["rounds"])
    keep = (rounds >= lo) & (rounds <= hi)
    rw = tuple(int(n) for n in rounds[keep])
    tests: dict = {k: {} for k in D.keys}
    obs_rows, pred_rows, where, shots = [], [], [], set()
    for j in D.counts["jobs"].values():
        key = f"{j['day']} r{j['rep']}"
        ob = {
            b: {
                p: wls_curve(rw, np.array(j["observed"][b][p])[keep], j["shots"])
                for p in j["patches"]
            }
            for b in "XZ"
        }
        fs = {}
        for src in (PRED, AFTER, *FROZEN):
            s = j["predicted"][src]
            fs[src] = FT.fits_struct(
                {
                    b: {
                        p: wls_curve(
                            rw, np.round(np.array(s["P"][b][p])[keep] * s["shots"]), s["shots"]
                        )
                        for p in j["patches"]
                    }
                    for b in "XZ"
                },
                "wls",
            )
        sl = FT.slopes(fs[FROZEN[0]], fs[FROZEN[1]], fs[PRED], j["f"])
        tests[key].update(
            FT.agreement(FT.fits_struct(ob, "wls"), fs[PRED], fs[AFTER], sl, j["f_uncertainty"])
        )
        s = j["predicted"][PRED]
        shots.add((j["shots"], s["shots"]))
        for b in "XZ":
            for p in j["patches"]:
                obs_rows.append(np.array(j["observed"][b][p], float)[keep])
                pred_rows.append(np.round(np.array(s["P"][b][p])[keep] * s["shots"]))
                where.append((key, b, p))
    assert len(shots) == 1
    ((n_obs, n_pred),) = shots
    for t in tests.values():
        for b in "XZ":
            if f"{b} lambda" not in t:
                t[f"{b} lambda"] = FT.ratio_test(t[f"{b} eps3"], t[f"{b} eps5"])
    out: dict = {"rounds": list(rw)}
    for q in TESTS:
        z = np.array(
            [(t[q]["observed"] - t[q]["predicted"]) / (t[q]["bound"] / 2) for t in tests.values()]
        )
        ok = np.isfinite(z)
        ratio = np.array([t[q]["observed"] / t[q]["predicted"] for t in tests.values()])
        out[q] = {
            "chi2": float((z[ok] ** 2).sum()),
            "repetitions": int(ok.sum()),
            "p": float(stats.chi2.sf((z[ok] ** 2).sum(), int(ok.sum()))),
            "passes": bool((z[ok] ** 2).sum() < stats.chi2.ppf(0.95, int(ok.sum()))),
            "agree": int((np.abs(z[ok]) < 2).sum()),
            "wls_O_over_M": t_interval(np.log(ratio[ok])),
        }
    # binomial point estimates, both sides fitted alike, no points dropped
    n = np.array(rw, float)
    eo = F.mle_batch(np.array(obs_rows), n_obs, n)[0]
    ep = F.mle_batch(np.array(pred_rows), n_pred, n)[0]
    got = {w: (o, m) for w, o, m in zip(where, eo, ep)}
    for q in TESTS:
        b, name = q.split()
        r = []
        for k in D.keys:
            o3, m3 = got[(k, b, "3")]
            o5, m5 = got[(k, b, "5")]
            r.append({"eps3": o3 / m3, "eps5": o5 / m5, "lambda": (o3 / o5) / (m3 / m5)}[name])
        out[q]["binomial_O_over_M"] = t_interval(np.log(r))
        out[q]["binomial_O_over_M_per_repetition"] = floats(r)
    return out


def window(D: Data) -> dict:
    per_rep = D.fits["per_rep"]
    shift: dict = {}
    for fit in ("wls", "binomial"):
        full = np.array([value(per_rep[k]["X"]["lambda"], fit) for k in D.keys])
        late = np.array([per_rep[k]["X"]["from_round_2"][fit] for k in D.keys])
        d = late - full
        shift[fit] = {
            "per_repetition": floats(d),
            "mean": float(d.mean()),
            "se_from_the_spread": float(d.std(ddof=1) / sqrt(len(d))),
            **{
                f"placement {p}": {
                    "mean": float(d[D.mask(p)].mean()),
                    "se_from_the_spread": float(d[D.mask(p)].std(ddof=1) / sqrt(D.mask(p).sum())),
                }
                for p in PLACEMENTS
            },
        }
    refit = {w: window_test(D, lo, hi) for w, (lo, hi) in WINDOWS.items()}
    for q in TESTS:  # the 1-8 refit is the prespecified test
        assert abs(refit["1-8"][q]["chi2"] - D.combined["model"][q]["chi2"]) < 1e-6
    for k, r in zip(D.keys, refit["1-8"]["X eps3"]["binomial_O_over_M_per_repetition"]):
        x = per_rep[k]["X"]
        # the binomial fit stops at an optimizer tolerance and repeats to about 1e-7 (README)
        assert abs(r - x["eps3"]["binomial"] / x["predicted"]["eps3"]["binomial"]) < 1e-6
    return {
        "lambda_X_rounds_2_8_minus_1_8": shift,
        "by_window": refit,
        "note": "WL tests refitted from results/fits_counts.json exactly as prespecified, f term from "
        "the frozen f = 0 and f = 1 predictions refitted in the same window. Binomial O/M: point "
        "estimates, both sides fitted alike; Student t interval over the ten repetitions.",
    }


# ----------------------------------------------------------------------------- selection
def selection(D: Data) -> dict:
    out: dict = {
        "what": "excess per round O - M in percentage points and the factor O/M, per patch "
        "(plain mean or geometric mean over the five days, t 4 dof). An equal additive excess "
        "gives a larger factor at the smaller eps, so only the additive scale and the "
        "fixed-against-reselected comparison speak to selection."
    }
    for fit in FITS:
        blk: dict = {}
        for b in "XZ":
            e = {}
            for d in "35":
                s = D.terms(fit, f"{b} eps{d}")
                e[d] = s
                for p in PLACEMENTS:
                    m = D.mask(p)
                    x = 100 * (s.O[m] - s.M[m])
                    t = float(stats.t.ppf(0.975, len(x) - 1))
                    se = float(x.std(ddof=1) / sqrt(len(x)))
                    blk[f"{b} d{d} placement {p}"] = {
                        "excess_pp": float(x.mean()),
                        "excess_pp_ci95": [float(x.mean() - t * se), float(x.mean() + t * se)],
                        "factor": t_interval(np.log(s.O[m] / s.M[m])),
                    }
            for p in PLACEMENTS:  # d = 5 minus d = 3, paired by day
                m = D.mask(p)
                x = 100 * ((e["5"].O[m] - e["5"].M[m]) - (e["3"].O[m] - e["3"].M[m]))
                t = float(stats.t.ppf(0.975, len(x) - 1))
                se = float(x.std(ddof=1) / sqrt(len(x)))
                blk[f"{b} placement {p}: excess d5 - d3 (pp)"] = {
                    "mean": float(x.mean()),
                    "ci95": [float(x.mean() - t * se), float(x.mean() + t * se)],
                }
            for d in "35":  # placement 2 against placement 1: re-selected d = 3, both d = 5 fixed
                y1 = np.log(e[d].O[D.mask(1)] / e[d].M[D.mask(1)])
                y2 = np.log(e[d].O[D.mask(2)] / e[d].M[D.mask(2)])
                res = stats.ttest_ind(y2, y1, equal_var=False)
                ci = res.confidence_interval(0.95)
                blk[f"{b} d{d}: factor placement 2 / placement 1"] = {
                    "ratio": float(exp(y2.mean() - y1.mean())),
                    "ci95": [float(exp(ci.low)), float(exp(ci.high))],
                    "welch_p": float(res.pvalue),
                }
        out[fit] = blk
    return out


# ---------------------------------------------------------------------------------- pool
def pool(D: Data) -> dict:
    px = D.combined["pooled_x"]
    return {
        "source": "runs/combined.json pooled_x (the prespecified code)",
        "Lambda_X": px["lambda"],
        "se": px["lambda_uncertainty"],
        "chi2_per_dof": px["spread"],
        "predicted_inverse_variance_weighted": px["predicted"],
        "observed_over_predicted": px["lambda"] / px["predicted"],
        "verdict": px["verdict"],
    }


# ----------------------------------------------------------------------------- headline
def headline(res: dict) -> dict:
    """The numbers the paper quotes, gathered from the sections above."""
    r, c = res["ratios"]["binomial"], res["chi2"]
    pw = res["power"]["wls"]
    pj = res["prejob"]["binomial"]
    win = res["window"]["by_window"]
    sel = res["selection"]["binomial"]

    def ci(v: dict) -> list:
        return [v["ratio"], *v["ci95"]]

    return {
        "O_over_M (binomial) [ratio, 95% low, high]": {
            q: {
                **{f"placement {p}": ci(r[q][f"placement {p}"]) for p in PLACEMENTS},
                "all ten": ci(r[q]["all ten"]),
                "tau": r[q]["all ten"]["tau"],
            }
            for q in TESTS
        },
        "prespecified chi2 (WL) [chi2, p, Holm p over the three X tests]": {
            q: [
                c["wls"][q]["chi2"],
                c["wls"][q]["p"],
                c["wls"][q].get("p_holm_over_the_three_X_tests"),
            ]
            for q in TESTS
        },
        "bias and scatter (WL) [bias chi2, p, scatter chi2, p]": {
            q: [c["wls"][q][k][x] for k in ("bias", "scatter") for x in ("chi2", "p")]
            for q in TESTS
        },
        "per placement chi2 (WL, 5 dof) [P1, P2]": {
            q: [c["wls"][q]["per_placement"][f"placement {p}"]["chi2"] for p in PLACEMENTS]
            for q in X_TESTS
        },
        "without the f term (WL) [chi2, agree; agree with it]": {
            q: [
                c["wls"][q]["without_f_term"]["chi2"],
                c["wls"][q]["without_f_term"]["agree"],
                c["wls"][q]["agree"],
            ]
            for q in X_TESTS
        },
        "placement 2 Lambda_X from_f range against median sigma_O (WL)": [
            res["budget"]["wls"]["X lambda"]["placement 2"]["from_f_range"],
            res["budget"]["wls"]["X lambda"]["placement 2"]["median_sigma_O"],
        ],
        "power (WL bounds) [Lambda_X at 0.06, Lambda_X found at 80%, eps3 and eps5 relative at 80%]": [
            pw["X lambda"]["realized"]["absolute"]["0.06"],
            pw["X lambda"]["realized"]["absolute_found_at_80pct"],
            pw["X eps3"]["realized"]["relative_found_at_80pct"],
            pw["X eps5"]["realized"]["relative_found_at_80pct"],
        ],
        "tracking, placement 1 Lambda_X": {
            fit: {
                "constant": [t["constant"]["chi2"], t["constant"]["p"]],
                "follows_prediction": [
                    t["follows_prediction"]["chi2"],
                    t["follows_prediction"]["p"],
                ],
                "pearson_r": t["pearson"]["r"],
                "without day 2 (3 dof)": [
                    t["leave_one_day_out"]["without day 2"]["follows_prediction"]["chi2"],
                    t["leave_one_day_out"]["without day 2"]["follows_prediction"]["p"],
                ],
            }
            for fit, t in ((f, res["tracking"][f]) for f in FITS)
        },
        "prejob (binomial)": {
            "X eps3 above the f = 1 prediction": pj["X eps3"]["observed_above_f1"],
            "X eps5 above the f = 1 prediction": pj["X eps5"]["observed_above_f1"],
            "jobs with measured f above 1": res["prejob"]["jobs_with_measured_f_above_1"],
            "Lambda_X bracket width, placement 1": pj["X lambda"]["bracket_width_range"][
                "placement 1"
            ],
            "Z max |M(f=1) - M(f=0)| [eps3, eps5, Lambda]": [
                pj["Z memory (does not depend on f)"][q]["max_abs_f1_minus_f0"]
                for q in ("Z eps3", "Z eps5", "Z lambda")
            ],
        },
        "window": {
            "Lambda_X 2-8 minus 1-8, WL [mean, se]": [
                res["window"]["lambda_X_rounds_2_8_minus_1_8"]["wls"]["mean"],
                res["window"]["lambda_X_rounds_2_8_minus_1_8"]["wls"]["se_from_the_spread"],
            ],
            "X eps3 chi2 by window": {w: v["X eps3"]["chi2"] for w, v in win.items()},
            "windows where X eps5 passes": [w for w, v in win.items() if v["X eps5"]["passes"]],
            "windows where X lambda passes": [w for w, v in win.items() if v["X lambda"]["passes"]],
            "binomial X eps5 O/M range over windows": [
                min(v["X eps5"]["binomial_O_over_M"]["ratio"] for v in win.values()),
                max(v["X eps5"]["binomial_O_over_M"]["ratio"] for v in win.values()),
            ],
        },
        "selection (binomial)": {
            "X excess pp [d3 P1, d3 P2, d5 P1, d5 P2]": [
                sel[f"X d{d} placement {p}"]["excess_pp"] for d in "35" for p in PLACEMENTS
            ],
            "X d3 factor, placement 2 / placement 1": sel["X d3: factor placement 2 / placement 1"],
        },
        "errors": {
            "Lambda_Z WL bootstrap sd / sigma [median, min, max]": res["errors"]["lambda"][
                "Lambda_Z"
            ]["wls_bootstrap_sd_over_sigma [median, min, max]"],
            "Z eps5 WL bootstrap sd / sigma [median, min, max]": res["errors"]["per_curve_summary"][
                "Z eps5"
            ]["wls_bootstrap_sd_over_sigma"],
            "WL Lambda_Z chi2 with the bootstrap sigma_O": res["errors"][
                "wls_with_bootstrap_sigma"
            ]["Z lambda"]["chi2_with_bootstrap_sigma_O"],
        },
        "table: largest inverse-variance minus plain mean": res["table"]["largest_iv_minus_plain"][
            0
        ],
        "pool": res["pool"],
    }


def main() -> None:
    D = Data()
    res = {
        "ratios": ratios(D),
        "chi2": chi2_tests(D),
        "budget": budget(D),
        "power": power(D),
        "tracking": tracking(D),
        "prejob": prejob(D),
        "table": table(D),
        "errors": errors(D),
        "window": window(D),
        "selection": selection(D),
        "pool": pool(D),
    }
    res = {"headline": headline(res), **res}
    h = res["headline"]
    for q, v in h["O_over_M (binomial) [ratio, 95% low, high]"].items():
        a = v["all ten"]
        print(f"O/M {q:9s} {a[0]:.3f} ({a[1]:.3f}-{a[2]:.3f}), tau {v['tau']:.3f}")
    for q in X_TESTS:
        x, p, ph = h["prespecified chi2 (WL) [chi2, p, Holm p over the three X tests]"][q]
        print(f"chi2 {q:9s} {x:6.2f}  p {p:.2g}  Holm {ph:.2g}")
    t = h["tracking, placement 1 Lambda_X"]["binomial"]
    pw = h[
        "power (WL bounds) [Lambda_X at 0.06, Lambda_X found at 80%, eps3 and eps5 relative at 80%]"
    ]
    print(
        f"placement 1 Lambda_X: constant chi2 {t['constant'][0]:.1f}, follows prediction "
        f"{t['follows_prediction'][0]:.1f}, r {t['pearson_r']:+.2f}; power at 0.06 {pw[0]:.2f}, "
        f"80% at {pw[1]:.3f}"
    )
    write_json("effect_sizes.json", res)


if __name__ == "__main__":
    main()
