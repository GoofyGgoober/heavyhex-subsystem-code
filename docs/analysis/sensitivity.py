"""How the prespecified agreement test depends on choices fixed before the run, and what its
tests could detect (Sec. agreement, Sec. days, App. stats).

  sigma_f      the standard error of the job's median f from the idle test's noise alone: a
               parametric bootstrap of each job's idle test (every data qubit's <X> and <Y> at
               every idle length redrawn from its shot counts, refitted by run._fit_share, the
               median taken over the qubits), beside the prespecified 1.25 s_f / sqrt(n), which
               the spread of f between qubits sets. The X tests with each: chi2, agreeing
               repetitions, per placement (5 dof), offset and scatter, a t test on the ten z,
               and O = k M with one k for all ten (9 dof); the f term's share of each variance
  monte_carlo  the predictions' Monte Carlo noise: the stored predicted curves at the measured
               f, f = 0 and f = 1 (200,000 shots per circuit) redrawn binomially and refitted,
               dM/df and sigma_sim rebuilt as run._slopes and run._agreement do, the X tests
               recomputed (the sd is that of one simulation; two independent simulations differ
               by sqrt(2) sd); the same with dM/df from the line through f = 0 and f = 1 for
               every job; the control rerun of idle_f_reanalyze.py, an actual re-simulation at
               the measured f; and a check that the three curves of a job, though seeded alike,
               carry independent noise
  windows      Lambda_X of device and model under six windows of rounds and both fits, plain
               means per placement: the shifts from the prespecified choice (WL, rounds 1-8)
               with the device's standard error over the repetitions, and a systematic error,
               half the range over the choices other than rounds 3-8 (there the distance-5
               curve is at P(n) = 0.38-0.51 throughout, and Lambda_X mostly noise)
  intervals    random-effects O/M with the Hartung-Knapp interval and with its variance factor
               floored at one; the common offset as a t test on the ten z
  power        each X test's power against a uniform miss at the bounds actually used, with the
               noise-only sigma_f and without the f term, beside the design study (power.json)
               and when docs/power.py and power.json entered the repository
  tracking     placement 1's Lambda_X against a constant multiple of its prediction, with each
               day left out; the X-stabilizer detection fractions of placement 1's distance-3
               patch over the bulk rounds by day, device against model
  scatter      the power of the exploratory calibration-scatter test,
               sigma_M'^2 = sigma_M^2 + (s_Q M)^2 (model_vs_device.py)

A repetition is one day and placement; repetition 1 is placement 1, whose two patches ran in one
job, and placement 2's patches ran in two jobs. Reads runs/ and results/ (combined.json,
fits.json, fits_counts.json, idle_f_inputs.json, model_vs_device.json, power.json, the control
rerun); seeded; about a minute in one process and well under 1 GB.

    python docs/analysis/sensitivity.py      # writes results/sensitivity.json
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from math import sqrt

import numpy as np
from scipy import stats
from scipy.optimize import brentq

from heavyhex.circuits.flagged import qubit_roles
from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import ROUNDS, SETTINGS
from heavyhex.patches.operators import build_operators

import effect_sizes as ES
import fits as FT
import fitting as F
from common import REPO, RUNS, Job, day_number, memory_index, read, result, write_json

X_TESTS = ("X lambda", "X eps3", "X eps5")
PLACEMENTS = (1, 2)
FITS = ("wls", "binomial")
PRED, F0, F1 = "predicted_at_measured_dephasing", "frozen_f0.0", "frozen_f1.0"
SIM = R.SIMULATED_SHOTS
B = 2000  # bootstrap and Monte Carlo replicates
SEED = 1
CRIT = float(stats.chi2.ppf(0.95, 10))
N = np.array(ROUNDS, float)


def floats(x) -> list:
    return [float(v) for v in x]


def spread(x) -> dict:
    x = np.asarray(x, float)
    lo, med, hi = np.percentile(x, [5, 50, 95])
    return {"mean": float(x.mean()), "sd": float(x.std(ddof=1)), "p5_p50_p95": [lo, med, hi]}


# ----------------------------------------------------------------------------------- data
class Data:
    def __init__(self) -> None:
        self.combined = read(RUNS / "combined.json")
        self.reps = self.combined["reps"]  # day, then repetition
        self.keys = [f"{r['day']} r{r['rep']}" for r in self.reps]
        self.day = np.array([day_number(r["day"]) for r in self.reps])
        self.placement = np.array([r["rep"] for r in self.reps])
        self.counts = result("fits_counts.json")
        fits = result("fits.json")
        self.per_rep = fits["per_rep"]
        self.tests = {
            "wls": [r["tests"] for r in self.reps],  # the prespecified analysis itself
            "binomial": [fits["agreement_tests"]["binomial"][k] for k in self.keys],
        }
        self.analysis = {name: read(RUNS / name / "analysis.json") for name in self.counts["jobs"]}
        for t in self.tests["wls"]:  # IBM did not recalibrate during any job
            assert all(t[q]["calibration_gap"] == 0 for q in X_TESTS)

    def job(self, i: int, d: str) -> str:
        """The job that ran repetition i's distance-d patch."""
        folders = self.reps[i]["folders"]
        return folders[0] if len(folders) == 1 else next(f for f in folders if f.endswith(f"d{d}"))

    def mask(self, p: int) -> np.ndarray:
        return self.placement == p


def terms(rows: list[dict], q: str) -> tuple[np.ndarray, ...]:
    """O, M, sigma_O, sigma_M (sim, f and calibration-gap terms) over the repetitions."""

    def get(key: str) -> np.ndarray:
        return np.array([r[q][key] for r in rows], float)

    sM = np.sqrt(get("simulator_noise") ** 2 + get("from_f") ** 2 + get("calibration_gap") ** 2 / 2)
    return get("observed"), get("predicted"), get("observed_uncertainty"), sM


def scaled(D: Data, fit: str, scale: dict[str, float]) -> list[dict]:
    """Each repetition's X tests with every job's f term multiplied by scale[job].

    A split repetition's Lambda adds its two jobs' relative terms in quadrature
    (run._ratio_test); a paired one scales Lambda's own term.
    """
    out = []
    for i, t in enumerate(D.tests[fit]):
        j3, j5 = D.job(i, "3"), D.job(i, "5")
        e3 = {**t["X eps3"], "from_f": t["X eps3"]["from_f"] * scale[j3]}
        e5 = {**t["X eps5"], "from_f": t["X eps5"]["from_f"] * scale[j5]}
        lam = dict(t["X lambda"])
        if j3 == j5:
            lam["from_f"] *= scale[j3]
        else:
            lam["from_f"] = FT.ratio_test(e3, e5)["from_f"]
        out.append({"X eps3": e3, "X eps5": e5, "X lambda": lam})
    return out


def summarize(D: Data, O, M, sO, sM) -> dict:
    """The agreement test and what else the ten z say."""
    z = (O - M) / np.hypot(sO, sM)
    zbar = float(z.mean())
    t = stats.ttest_1samp(z, 0.0)
    multiple = ES.track(O, M, sO, sM)["follows_prediction"]
    return {
        **ES.chi2_p((z**2).sum(), len(z)),
        "agree": int((np.abs(z) < 2).sum()),
        "z": floats(z),
        "per_placement": {
            f"placement {p}": {
                **ES.chi2_p((z[D.mask(p)] ** 2).sum(), int(D.mask(p).sum())),
                "agree": int((np.abs(z[D.mask(p)]) < 2).sum()),
            }
            for p in PLACEMENTS
        },
        "offset": ES.chi2_p(len(z) * zbar**2, 1),
        "scatter": ES.chi2_p(((z - zbar) ** 2).sum(), len(z) - 1),
        "offset_t_test": {
            "mean_z": zbar,
            "t": float(t.statistic),
            "dof": len(z) - 1,
            "p": float(t.pvalue),
        },
        "constant_multiple (O = k M, 9 dof)": {
            "k": multiple["c"],
            "chi2": multiple["chi2"],
            "p": multiple["p"],
        },
    }


# ------------------------------------------------------------------------------- sigma_f
def share_fit(r, log_x0, log_x1, shots) -> tuple[np.ndarray, np.ndarray]:
    """run._fit_share over leading axes (the last axis is the idle length); NaN as there."""
    keep = r > 0.05
    extra = np.where(keep, log_x0 - np.log(np.where(keep, r, 1.0)), 0.0)
    full = np.where(keep, log_x0 - log_x1, 0.0)
    weight = np.where(keep, shots * r**2 / (1 - np.minimum(r, 0.999) ** 2), 0.0)
    m = keep.sum(-1)
    w, wx, wy = weight.sum(-1), (weight * full).sum(-1), (weight * extra).sum(-1)
    det = w * (weight * full**2).sum(-1) - wx**2
    with np.errstate(divide="ignore", invalid="ignore"):
        many = m >= 2
        share = np.where(many, (w * (weight * full * extra).sum(-1) - wx * wy) / det, wy / wx)
        sigma = np.where(many, np.sqrt(w / det), np.sqrt(w) / wx)
    bad = (m == 0) | (many & (det <= 0)) | (~many & (wx <= 0)) | ~(sigma <= R.LOOSEST_F)
    return np.where(bad, np.nan, share), np.where(bad, np.nan, sigma)


def idle_r(x, y, nx, ny):
    """sqrt(<X>^2 + <Y>^2) with the shot noise taken out, as run._idle_dephasing."""
    r2 = x**2 + y**2 - (1 - x**2) / nx - (1 - y**2) / ny
    return np.sqrt(np.maximum(r2, 0))


def redraw(mean, shots, rng: np.random.Generator) -> np.ndarray:
    """B redraws of <X> (or <Y>), the mean of shots +-1 outcomes."""
    return 2 * rng.binomial(shots.astype(int), (1 + mean) / 2, size=(B, *mean.shape)) / shots - 1


def sigma_f_bootstrap(D: Data, rng: np.random.Generator) -> dict:
    """Each job's sigma_f from the idle test's noise alone, beside the prespecified one."""
    inputs = result("idle_f_inputs.json")
    out = {}
    for name, job in inputs.items():
        idle = D.analysis[name]["idle_test"]
        q = list(job["qubits"].values())
        x, y = (np.array([v[k] for v in q]) for k in ("x", "y"))
        nx, ny = (np.array([v[k] for v in q], float) for k in ("nx", "ny"))
        l0, l1 = (np.array([v[k] for v in q]) for k in ("log_x0", "log_x1"))
        shots = (nx + ny) / 2
        share, _ = share_fit(idle_r(x, y, nx, ny), l0, l1, shots)
        stored = [
            idle["per_qubit"][key.split("/")[0]][int(key.split("/")[1])] for key in job["qubits"]
        ]
        stored = np.array([np.nan if s is None else s for s in stored], float)
        assert np.array_equal(np.isnan(share), np.isnan(stored))
        assert np.nanmax(np.abs(share - stored)) < 1e-9
        ok = np.isfinite(share)
        assert abs(np.median(share[ok]) - idle["median"]) < 1e-12
        sf = 1.25 * np.std(share[ok]) / sqrt(ok.sum())
        assert abs(sf - idle["median_uncertainty"]) < 1e-12
        xb, yb = redraw(x, nx, rng), redraw(y, ny, rng)
        sb, _ = share_fit(idle_r(xb, yb, nx, ny), l0, l1, shots)
        medians = np.nanmedian(sb, axis=1)
        noise = float(medians.std(ddof=1))
        out[name] = {
            "qubits": int(ok.sum()),
            "f": idle["median"],
            "sigma_f_prespecified": idle["median_uncertainty"],
            "sigma_f_idle_noise": noise,
            "ratio": noise / idle["median_uncertainty"],
            "bootstrap_mean_minus_f": float(medians.mean() - idle["median"]),
            "uniformity_chi2_per_dof": idle["uniformity"]["chi2_per_dof"],
        }
    return out


def scales(boot: dict) -> dict[str, dict[str, float]]:
    """Each job's f term multiplied by 1 (as prespecified), by the noise-only ratio, or by 0."""
    return {
        "prespecified": dict.fromkeys(boot, 1.0),
        "idle_noise_only": {j: v["ratio"] for j, v in boot.items()},
        "no_f_term": dict.fromkeys(boot, 0.0),
    }


def sigma_f_section(D: Data, boot: dict) -> dict:
    noise = [v["sigma_f_idle_noise"] for v in boot.values()]
    pre = [v["sigma_f_prespecified"] for v in boot.values()]
    out: dict = {
        "per_job": boot,
        "sigma_f_range": {
            "prespecified": [min(pre), max(pre)],
            "idle_noise_only": [min(noise), max(noise)],
        },
        "ratio_range": [
            min(v["ratio"] for v in boot.values()),
            max(v["ratio"] for v in boot.values()),
        ],
    }
    for fit in FITS:
        out[fit] = {}
        for label, scale in scales(boot).items():
            rows = scaled(D, fit, scale)
            out[fit][label] = {q: summarize(D, *terms(rows, q)) for q in X_TESTS}
        # the check: scale 1 is the stored test
        for q in X_TESTS:
            ref = ES.chi2_p(
                sum(
                    ((t[q]["observed"] - t[q]["predicted"]) / (t[q]["bound"] / 2)) ** 2
                    for t in D.tests[fit]
                ),
                10,
            )
            assert abs(out[fit]["prespecified"][q]["chi2"] - ref["chi2"]) < 1e-9
    share: dict = {}
    for q in X_TESTS:
        _, _, sO, sM = terms(D.tests["wls"], q)
        ff = np.array([abs(t[q]["from_f"]) for t in D.tests["wls"]])
        s = ff**2 / (sO**2 + sM**2)
        share[q] = {
            f"placement {p}": [float(s[D.mask(p)].min()), float(s[D.mask(p)].max())]
            for p in PLACEMENTS
        }
        share[q]["all"] = [float(s.min()), float(s.max())]
    out["f_term_share_of_the_z_variance (WL, prespecified) [min, max]"] = share
    return out


# --------------------------------------------------------------------------- Monte Carlo
def slope_weights(f: float) -> np.ndarray:
    """dM/df at f as weights on M(0), M(f), M(1): run._slopes's parabola, or its line."""
    if 0.05 < f < 0.95:
        return np.array([(f - 1) / f, (2 * f - 1) / (f * (f - 1)), f / (1 - f)])
    return np.array([-1.0, 0.0, 1.0])


def wls_curves(P, rng: np.random.Generator | None) -> tuple[np.ndarray, np.ndarray]:
    """eps and its widened se of a stored predicted curve, (B,) redrawn, or (1,) as stored."""
    k = np.round(np.asarray(P) * SIM)
    if rng is not None:
        k = rng.binomial(SIM, k / SIM, size=(B, len(k)))
    return F.wls_batch(N, np.atleast_2d(k) / SIM, SIM)


def job_predictions(D: Data, rng: np.random.Generator | None) -> dict:
    """Per job: X eps (and Lambda when both patches ran) at f = 0, the measured f and f = 1,
    each (value, sim se), as arrays over replicates."""
    out = {}
    for name, j in D.counts["jobs"].items():
        at = {}
        for src in (F0, PRED, F1):
            P = j["predicted"][src]["P"]["X"]
            at[src] = {f"eps{p}": wls_curves(P[p], rng) for p in j["patches"]}
            if len(j["patches"]) == 2:
                (e3, s3), (e5, s5) = at[src]["eps3"], at[src]["eps5"]
                lam = e3 / e5
                at[src]["lambda"] = (lam, lam * np.sqrt((s3 / e3) ** 2 + (s5 / e5) ** 2))
        w = slope_weights(j["f"])
        out[name] = {
            q: {
                "M": at[PRED][q][0],
                "sim": at[PRED][q][1],
                "slope": sum(wk * at[src][q][0] for wk, src in zip(w, (F0, PRED, F1))),
                "slope_line": at[F1][q][0] - at[F0][q][0],
            }
            for q in at[PRED]
        }
    return out


def mc_z2(D: Data, pred: dict, sigma_f: dict, line: bool = False) -> dict[str, np.ndarray]:
    """z^2 of each X test from (redrawn) predictions, repetitions x replicates."""
    key = "slope_line" if line else "slope"
    z2: dict = {q: [] for q in X_TESTS}
    for i, t in enumerate(D.tests["wls"]):
        j3, j5 = D.job(i, "3"), D.job(i, "5")
        e3, e5 = pred[j3]["eps3"], pred[j5]["eps5"]
        rows = {
            "X eps3": (e3["M"], e3["sim"], e3[key] * sigma_f[j3]),
            "X eps5": (e5["M"], e5["sim"], e5[key] * sigma_f[j5]),
        }
        if j3 == j5:
            lam = pred[j3]["lambda"]
            rows["X lambda"] = (lam["M"], lam["sim"], lam[key] * sigma_f[j3])
        else:
            (m3, n3, f3), (m5, n5, f5) = rows["X eps3"], rows["X eps5"]
            m = m3 / m5
            rows["X lambda"] = (
                m,
                m * np.sqrt((n3 / m3) ** 2 + (n5 / m5) ** 2),
                m * np.sqrt((f3 / m3) ** 2 + (f5 / m5) ** 2),
            )
        for q, (M, sim, ff) in rows.items():
            O, sO = t[q]["observed"], t[q]["observed_uncertainty"]
            z2[q].append((O - M) ** 2 / (sO**2 + sim**2 + ff**2))
    return {q: np.array(v) for q, v in z2.items()}


def seeds_check() -> dict:
    """run.predict seeds every circuit by its round count, so the curves at f = 0, f and 1 share
    a seed. Does that correlate their noise? 20,000 shots of one circuit at two f values."""
    job = Job(RUNS / "ibm_fez-2026-10-05-r2-d5")
    idx = memory_index("X", 4)
    failed = {}
    for f in (job.f, 1.0):
        model, _, _ = job.model(idx, f=f)
        det, flips = model.compile_detector_sampler(seed=SETTINGS[idx].rounds).sample(
            20_000, separate_observables=True
        )
        failed[f] = (R._decode(model, det) != flips)[:, 0], det
    (a, da), (b, db) = failed.values()
    corr = [
        np.corrcoef(da[:, k], db[:, k])[0, 1]
        for k in range(da.shape[1])
        if da[:, k].std() > 0 and db[:, k].std() > 0
    ]
    return {
        "job": job.name,
        "setting": str(SETTINGS[idx]),
        "f": [job.f, 1.0],
        "shots": 20_000,
        "logical_failure_correlation": float(np.corrcoef(a, b)[0, 1]),
        "mean_detector_correlation": float(np.mean(corr)),
    }


def monte_carlo(D: Data, boot: dict, rng: np.random.Generator) -> dict:
    variants = {
        "prespecified": {j: v["sigma_f_prespecified"] for j, v in boot.items()},
        "idle_noise_only": {j: v["sigma_f_idle_noise"] for j, v in boot.items()},
    }
    stored = job_predictions(D, None)
    redrawn = job_predictions(D, rng)
    out: dict = {
        "what": f"{B} redraws of every stored predicted curve (binomial, {SIM} shots per point), "
        "refitted (WL); the observations fixed. sd is the Monte Carlo spread a fresh simulation "
        "would give the chi2; the mean also carries the redraw's added noise."
    }
    bar5 = float(stats.chi2.ppf(0.95, 5))
    for label, sf in variants.items():
        blk: dict = {}
        for line in (False, True):
            st, mc = mc_z2(D, stored, sf, line), mc_z2(D, redrawn, sf, line)
            for q in X_TESTS:
                chi = mc[q].sum(0)
                values, counts = np.unique((mc[q] < 4).sum(0), return_counts=True)
                by = {p: mc[q][D.mask(p)].sum(0) for p in PLACEMENTS}
                blk.setdefault(q, {})["line_slope" if line else "prespecified_slope"] = {
                    "chi2_stored": float(st[q].sum()),
                    "agree_stored": int((st[q] < 4).sum()),
                    "chi2_redrawn": spread(chi),
                    "fails (chi2 > 18.31)": float((chi > CRIT).mean()),
                    "agree_redrawn": {int(v): float(c / B) for v, c in zip(values, counts)},
                    "per_placement_chi2_stored": {
                        f"placement {p}": float(st[q][D.mask(p)].sum()) for p in PLACEMENTS
                    },
                    "per_placement_chi2_redrawn": {
                        f"placement {p}": spread(v) for p, v in by.items()
                    },
                    "per_placement_fails (chi2 > 11.07)": {
                        f"placement {p}": float((v > bar5).mean()) for p, v in by.items()
                    },
                }
        out[label] = blk
    # the stored redo reproduces the prespecified test
    for q in X_TESTS:
        assert (
            abs(
                out["prespecified"][q]["prespecified_slope"]["chi2_stored"]
                - D.combined["model"][q]["chi2"]
            )
            < 1e-6
        )
    slopes = {}
    for name, j in D.counts["jobs"].items():
        sf = boot[name]["sigma_f_prespecified"]
        slopes[name] = {
            "f": j["f"],
            "rule": "parabola" if 0.05 < j["f"] < 0.95 else "line",
            **{
                q: {
                    "slope": float(stored[name][q]["slope"][0]),
                    "slope_mc_sd": float(redrawn[name][q]["slope"].std(ddof=1)),
                    "slope_line": float(stored[name][q]["slope_line"][0]),
                    "from_f": float(stored[name][q]["slope"][0] * sf),
                    "from_f_mc_sd": float(redrawn[name][q]["slope"].std(ddof=1) * sf),
                }
                for q in stored[name]
            },
        }
    out["slopes_by_job"] = slopes
    ctrl = result("idle_f_combined_control.json")
    out["control_rerun (idle_f_reanalyze.py --control: measured-f predictions simulated again)"] = {
        q: {
            "chi2": ctrl["model"][q]["chi2"],
            "agree": sum(
                abs(e["tests"][q]["observed"] - e["tests"][q]["predicted"]) < e["tests"][q]["bound"]
                for e in ctrl["reps"]
            ),
        }
        for q in X_TESTS
    }
    out["seeds_check"] = seeds_check()
    return out


# ------------------------------------------------------------------------------ windows
def in_window(lo: int, hi: int) -> np.ndarray:
    """The round counts n with lo <= n <= hi, as a mask over ROUNDS."""
    return np.array([lo <= n <= hi for n in ROUNDS])


def window_lambda(D: Data, lo: int, hi: int, fit: str) -> tuple[np.ndarray, np.ndarray]:
    """Lambda_X of device and model per repetition, both fitted on rounds lo..hi."""
    keep = in_window(lo, hi)
    n = N[keep]
    jobs = D.counts["jobs"]

    def eps(k, shots):
        k = np.asarray(k, float)[keep]
        if fit == "wls":
            return F.wls(n, k / shots, shots).eps
        return float(F.mle_batch(k[None, :], shots, n)[0][0])

    obs, pred = [], []
    for i in range(len(D.reps)):
        j3, j5 = jobs[D.job(i, "3")], jobs[D.job(i, "5")]
        o = eps(j3["observed"]["X"]["3"], j3["shots"]) / eps(j5["observed"]["X"]["5"], j5["shots"])
        p3 = np.round(np.array(j3["predicted"][PRED]["P"]["X"]["3"]) * SIM)
        p5 = np.round(np.array(j5["predicted"][PRED]["P"]["X"]["5"]) * SIM)
        obs.append(o)
        pred.append(eps(p3, SIM) / eps(p5, SIM))
    return np.array(obs), np.array(pred)


def windows(D: Data) -> dict:
    jobs = D.counts["jobs"]
    cells: dict = {}
    for fit in FITS:
        for w, (lo, hi) in ES.WINDOWS.items():
            o, m = window_lambda(D, lo, hi, fit)
            cell = {"device": floats(o), "model": floats(m)}
            if fit == "wls":  # the distance-5 points the WL fit keeps, P(n) < 1/2
                keep = in_window(lo, hi)
                used = []
                for i in range(len(D.reps)):
                    j5 = jobs[D.job(i, "5")]
                    k = np.array(j5["observed"]["X"]["5"])[keep]
                    used.append(int((k < j5["shots"] / 2).sum()))
                cell["wls_d5_points_used [min, max]"] = [min(used), max(used)]
            for label, mask in (
                ("placement 1", D.mask(1)),
                ("placement 2", D.mask(2)),
                ("all ten", None),
            ):
                oo, mm = (o, m) if mask is None else (o[mask], m[mask])
                cell[label] = {
                    "device": float(oo.mean()),
                    "model": float(mm.mean()),
                    "device_minus_model": float(oo.mean() - mm.mean()),
                }
            cells[f"{fit} {w}"] = cell
    base = cells["wls 1-8"]
    stored = np.array([t["X lambda"]["observed"] for t in D.tests["wls"]])
    assert np.allclose(base["device"], stored, rtol=0, atol=1e-12)
    binomial = np.array([D.per_rep[k]["X"]["lambda"]["binomial"] for k in D.keys])
    assert np.allclose(cells["binomial 1-8"]["device"], binomial, rtol=0, atol=1e-6)
    # Rounds 3-8 leave out the two rounds where the distance-5 curve is furthest from 1/2, so
    # their Lambda_X scatters by about 0.2 between repetitions and its shift is mostly noise; the
    # systematic error is half the range over the other ten choices.
    kept = [k for k in cells if not k.endswith("3-8")]
    masks = {"placement 1": D.mask(1), "placement 2": D.mask(2), "all ten": D.placement > 0}
    shifts: dict = {}
    for label, m in masks.items():
        blk: dict = {"from wls 1-8": {}}
        for k, c in cells.items():
            d = (np.array(c["device"]) - np.array(base["device"]))[m]
            blk["from wls 1-8"][k] = {
                **{x: c[label][x] - base[label][x] for x in ("device", "model")},
                "device_se_over_repetitions": float(d.std(ddof=1) / sqrt(len(d))),
            }
        for x in ("device", "model", "device_minus_model"):
            every = np.array([c[label][x] for c in cells.values()])
            some = np.array([cells[k][label][x] for k in kept])
            blk[x] = {
                "systematic (half range without 3-8)": float(np.ptp(some) / 2),
                "range without 3-8": [float(some.min()), float(some.max())],
                "half range, all 12": float(np.ptp(every) / 2),
            }
        shifts[label] = blk
    drop1: dict = {}
    for fit in FITS:
        late, full = cells[f"{fit} 2-8"], cells[f"{fit} 1-8"]
        for label, m in masks.items():
            d = (np.array(late["device"]) - np.array(full["device"]))[m]
            drop1.setdefault(fit, {})[label] = {
                **{x: late[label][x] - full[label][x] for x in ("device", "model")},
                "device_se_over_repetitions": float(d.std(ddof=1) / sqrt(len(d))),
            }
    return {
        "what": "plain means over the days of Lambda_X = eps3 / eps5, device and model (the "
        "prediction at the measured f, 200,000 shots), each curve fitted on the window",
        "by_choice": cells,
        "shifts": shifts,
        "leaving_out_round_1 (2-8 minus 1-8)": drop1,
    }


# ----------------------------------------------------------------------------- intervals
def intervals(D: Data, boot: dict) -> dict:
    out: dict = {}
    for fit in FITS:
        for label in ("prespecified", "idle_noise_only"):
            rows = scaled(D, fit, scales(boot)[label])
            blk = {}
            for q in X_TESTS:
                O, M, sO, sM = terms(rows, q)
                re = ES.random_effects(np.log(O / M), (sO / O) ** 2 + (sM / M) ** 2)
                blk[q] = {
                    "ratio": re["ratio"],
                    "hartung_knapp_ci95": re["ci95"],
                    "hartung_knapp_floored_ci95": re["ci95_factor_at_least_1"],
                    "tau": re["tau"],
                    "Q": re["Q"],
                }
            out.setdefault(fit, {})[label] = blk
        blk = {}
        for q in ("Z eps3", "Z eps5", "Z lambda"):
            O, M, sO, sM = terms(D.tests[fit], q)
            re = ES.random_effects(np.log(O / M), (sO / O) ** 2 + (sM / M) ** 2)
            blk[q] = {
                "ratio": re["ratio"],
                "hartung_knapp_ci95": re["ci95"],
                "hartung_knapp_floored_ci95": re["ci95_factor_at_least_1"],
            }
        out[fit]["prespecified"].update(blk)
    offset = {}
    for fit in FITS:
        for q in (*X_TESTS, "Z eps3", "Z eps5", "Z lambda"):
            O, M, sO, sM = terms(D.tests[fit], q)
            z = (O - M) / np.hypot(sO, sM)
            t = stats.ttest_1samp(z, 0.0)
            offset.setdefault(fit, {})[q] = {
                "mean_z": float(z.mean()),
                "unit_variance_offset (10 zbar^2, 1 dof)": ES.chi2_p(10 * z.mean() ** 2, 1),
                "t_test (9 dof)": {"t": float(t.statistic), "p": float(t.pvalue)},
            }
    out["offset"] = offset
    return out


# --------------------------------------------------------------------------------- power
def ncx2_power(nc: float) -> float:
    return float(stats.ncx2.sf(CRIT, 10, nc))


NC80 = brentq(lambda x: ncx2_power(x) - 0.8, 1e-3, 100.0)


def power_of(M, sig) -> dict:
    """Against a uniform miss: absolute in Lambda, relative to M for eps."""
    inv, rel = float((1 / sig**2).sum()), float(((M / sig) ** 2).sum())
    return {
        "median_sigma": float(np.median(sig)),
        "absolute": {f"{d:.2f}": ncx2_power(d * d * inv) for d in (0.06, 0.09, 0.13)},
        "absolute_found_at_80pct": sqrt(NC80 / inv),
        "relative": {f"{r:.2f}": ncx2_power(r * r * rel) for r in (0.10, 0.15, 0.20)},
        "relative_found_at_80pct": sqrt(NC80 / rel),
    }


def first_commit(path: str) -> dict | None:
    """The commit that added path, from git (None outside a clone)."""
    try:
        line = subprocess.run(
            ["git", "log", "--diff-filter=A", "--follow", "--format=%h %aI", "--", path],
            cwd=REPO,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.split("\n")[-2]
    except (OSError, subprocess.CalledProcessError, IndexError):
        return None
    commit, stamp = line.split()
    utc = datetime.fromisoformat(stamp).astimezone(timezone.utc)
    return {"commit": commit, "utc": utc.strftime("%Y-%m-%d %H:%M:%S")}


def power(D: Data, boot: dict) -> dict:
    out: dict = {
        "what": "power of chi2 = sum z^2 > 18.31 against the same miss in every repetition, "
        "noncentral chi2 with 10 dof at each repetition's sigma = sqrt(sigma_O^2 + sigma_M^2)",
    }
    for label, scale in scales(boot).items():
        rows = scaled(D, "wls", scale)
        out[label] = {}
        for q in X_TESTS:
            _, M, sO, sM = terms(rows, q)
            out[label][q] = power_of(M, np.hypot(sO, sM))
    planned = result("power.json")
    out["design_study (power.json)"] = {
        f"f = {f}": {
            "sigma_one_repetition": v["10 reps"]["sigma_one_rep"],
            "Lambda_miss_caught_80pct": v["10 reps"]["gap_caught_80pct"],
            "x_tests_fail_when_model_right": v["10 reps"]["x_tests_fail_when_right"],
        }
        for f, v in planned["f"].items()
    }
    created = min(read(RUNS / name / "job.json")["created_utc"] for name in D.counts["jobs"])
    out["provenance"] = {
        "docs/power.py first committed": first_commit("docs/power.py"),
        "docs/analysis/results/power.json first committed": first_commit(
            "docs/analysis/results/power.json"
        ),
        "first job created (IBM)": created,
        "inputs of power.json": "the frozen f = 0 and f = 1 predictions (prediction.json) of all "
        "fifteen jobs; days 2-5's were made on their own days",
        "assumes": "f known exactly (no f term) and sigma_O at the predicted curves",
    }
    return out


# ------------------------------------------------------------------------------ tracking
def stabilizer_rates(D: Data) -> dict:
    """Placement 1's distance-3 X stabilizers, 8-round X memory, bulk rounds t = 2..8: the
    detection fraction by day, device against the model at the measured f and at f = 0, 1."""
    idx = memory_index("X", 8)
    roles = qubit_roles(build_operators(3))
    out: dict = {}
    for i in np.flatnonzero(D.mask(1)):
        name = D.reps[i]["folders"][0]
        a = D.analysis[name]
        chip = read(RUNS / name / "run.json")["chips"]["3"]
        cal = read(RUNS / name / "calibration.json")
        frozen = read(RUNS / name / "prediction.json")
        shots = D.counts["jobs"][name]["shots"]
        sources = {
            "device": a["observed"]["settings"][idx],
            "model": a[PRED]["settings"][idx],
            "model_f0": frozen["0.0"]["settings"][idx],
            "model_f1": frozen["1.0"]["settings"][idx],
        }
        labels = [tuple(x) for x in sources["device"]["detectors"]]
        for s in sources.values():
            assert [tuple(x) for x in s["detectors"]] == labels
        day = {}
        for stab in dict.fromkeys(n for p, _, n in labels if p == "3"):
            cols = [
                k for k, (p, r, n) in enumerate(labels) if p == "3" and n == stab and 1 <= r <= 7
            ]
            rate = {
                k: float(np.mean([s["detector_rates"][c] for c in cols]))
                for k, s in sources.items()
            }
            p = np.array([sources["device"]["detector_rates"][c] for c in cols])
            data = [int(v) for v in stab.lstrip("X").split("X")]
            gauges = [
                g
                for g in roles.x_ancillas
                if {int(v) for v in g.lstrip("X").split("X")} <= set(data)
            ]
            ancillas = [chip[roles.x_ancillas[g]] for g in gauges]
            qubits = [chip[v - 1] for v in data]
            day[stab] = {
                **rate,
                "device_se": float(np.sqrt((p * (1 - p)).sum() / shots) / len(cols)),
                "device_over_model": rate["device"] / rate["model"],
                "ancillas": ancillas,
                "data": qubits,
                "ancilla_readout_error": [cal["readout"][str(q)] for q in ancillas],
                "data_t1_us": [cal["t1_us"][str(q)] for q in qubits],
                "data_t2_us": [cal["t2_us"][str(q)] for q in qubits],
            }
        lam = D.tests["wls"][i]["X lambda"]
        out[f"day {int(D.day[i])}"] = {
            "Lambda_X": {"device": lam["observed"], "model": lam["predicted"]},
            "stabilizers": day,
        }
    summary = {}
    for stab in out["day 1"]["stabilizers"]:
        by = {
            k: np.array([v["stabilizers"][stab][k] for v in out.values()])
            for k in ("device", "model")
        }
        summary[stab] = {
            **{f"sd over days, {k}": float(v.std(ddof=1)) for k, v in by.items()},
            **{
                f"day 2 minus the other days' mean, {k}": float(v[1] - np.delete(v, 1).mean())
                for k, v in by.items()
            },
            "pearson_device_model": float(np.corrcoef(by["device"], by["model"])[0, 1]),
        }
    out["over the days"] = summary
    return out


def tracking(D: Data) -> dict:
    out: dict = {}
    for fit in FITS:
        O, M, sO, sM = terms(D.tests[fit], "X lambda")
        m = D.mask(1)
        blk = {"all days": ES.track(O[m], M[m], sO[m], sM[m])}
        for i, d in enumerate(D.day[m]):
            k = np.arange(m.sum()) != i
            t = ES.track(O[m][k], M[m][k], sO[m][k], sM[m][k])
            blk[f"without day {d}"] = {
                "constant": t["constant"],
                "follows_prediction": t["follows_prediction"],
            }
        out[fit] = blk
    out["stabilizers (placement 1, distance 3)"] = stabilizer_rates(D)
    return out


# ------------------------------------------------------------------------------- scatter
def scatter_power(D: Data) -> dict:
    s = result("model_vs_device.json")["scatter"]
    out: dict = {
        "what": "the exploratory test's chi2 over ten repetitions, sigma'^2 = sigma^2 + (s_Q M)^2"
    }
    for q in X_TESTS:
        O, M, sO, sM = terms(D.tests["wls"], q)
        sig = np.sqrt(sO**2 + sM**2 + (s[q]["relative_scatter"] * M) ** 2)
        assert abs((((O - M) / sig) ** 2).sum() - s[q]["chi2_with_scatter"]) < 1e-9
        out[q] = {"relative_scatter": s[q]["relative_scatter"], **power_of(M, sig)}
    return out


# ---------------------------------------------------------------------------------- main
def main() -> None:
    rng = np.random.default_rng(SEED)
    D = Data()
    boot = sigma_f_bootstrap(D, rng)
    res = {
        "sigma_f": sigma_f_section(D, boot),
        "monte_carlo": monte_carlo(D, boot, rng),
        "windows": windows(D),
        "intervals": intervals(D, boot),
        "power": power(D, boot),
        "tracking": tracking(D),
        "scatter_power": scatter_power(D),
    }
    sf = res["sigma_f"]["wls"]
    for label in ("prespecified", "idle_noise_only", "no_f_term"):
        print(
            f"{label:16s} "
            + "  ".join(
                f"{q} {sf[label][q]['chi2']:.1f} ({sf[label][q]['agree']})" for q in X_TESTS
            )
        )
    for q in X_TESTS:
        mc = res["monte_carlo"]["prespecified"][q]["prespecified_slope"]
        print(f"MC {q:9s} chi2 {mc['chi2_stored']:.1f}, sd {mc['chi2_redrawn']['sd']:.1f}")
    write_json("sensitivity.json", res)


if __name__ == "__main__":
    main()
