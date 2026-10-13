"""Checks of the statistics behind the X-memory tests, from stored results only (Sec. results,
Sec. days, App. stats, the table of rules).

  binomial_primary  each pair's eps3, eps5 and Lambda_X with the binomial fit first and the
                    log-linear (prespecified) fit beside it: plain five-day means as in the
                    pairs' table, observed minus predicted, and the share of the log-linear
                    Lambda_X gap that the binomial fit removes; observed over predicted per pair
                    and over all ten pair-days; the three X tests under the binomial fit
  rule_size         the familywise size of the prespecified rule "reject if any of the three
                    chi2 exceeds chi2_{10,0.95} = 18.31": exactly for three independent tests, and
                    by simulation with z_Lambda = (z3 - z5)/sqrt(2) and with each pair-day's own
                    correlations of (z_Lambda, z3, z5), which follow from the terms of sigma_O and
                    sigma_M (Lambda = eps3/eps5 to first order; one f for both patches of pair 1,
                    one per job for pair 2); the single bound that would give a familywise 5%,
                    Bonferroni's, and Holm's p
  fit_window        the fit-window systematic on each pair's Lambda_X with and without rounds 3-8
  lambda_offset     observed over predicted Lambda_X over the ten pair-days (DerSimonian-Laird,
                    Hartung-Knapp; the paper quotes the interval floored at the inverse-variance
                    one) without day 2 of pair 1, without days 1 and 2 of pair 1, and with day 2's
                    prediction lowered as App. where does (day_swing.py); t test on the z values
  tracking          pair 1, Lambda_X = c M with one c over the days (effect_sizes.track): c and
                    chi2 with all five days and without day 2, chi2 at c = 1, and c = 1 against
                    the fitted c (Delta chi2, 1 dof)
  detector_bar      the detector side check's bar sqrt(2 ln N) beside the expected largest of N
                    standard normal values, its median and 95th percentile, and the number of
                    detectors beyond each, recounted from analysis.json (checked against the stored
                    count at the prespecified bar)
  f_read_off        the two prespecified read-offs of f from the memory circuits (from eps and
                    from the stabilizer detection rates) against the idle test

Reads runs/combined.json, the jobs' analysis.json and results/{fits, fits_counts, effect_sizes,
prespecified_tests, model_vs_device, day_swing}.json; a few seconds, one process.

    python docs/analysis/stat_checks.py      # writes results/stat_checks.json
"""

from __future__ import annotations

from math import log, sqrt

import numpy as np
from scipy import integrate, stats

from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import SETTINGS

import effect_sizes as ES
from common import jobs, result, write_json

FITS = ("binomial", "wls")  # binomial first: it keeps every point and is unbiased near P = 1/2
X_TESTS = ("X lambda", "X eps3", "X eps5")
BOUND = float(stats.chi2.ppf(0.95, 10))
REPS = 1_000_000  # simulated experiments for the rule's size
SEED = 5


def floats(x) -> list:
    return [float(v) for v in x]


# ------------------------------------------------------------------------ binomial primary
def binomial_primary(D: ES.Data) -> dict:
    cells = result("prespecified_tests.json")["table_pairs"]["cells"]
    es = result("effect_sizes.json")
    pt = result("prespecified_tests.json")
    out: dict = {
        "what": "plain means over each pair's five days (the pairs' table), observed and "
        "predicted at the measured f, each with its error as in that table; O/M per pair the "
        "geometric mean with Student t (4 dof), over all ten the DerSimonian-Laird mean with the "
        "Hartung-Knapp interval floored at the inverse-variance one",
    }
    for p in (1, 2):
        blk: dict = {}
        for q in ("eps3", "eps5", "lambda"):
            row: dict = {}
            for fit in FITS:
                c = cells[f"X pair {p} {fit} {q}"]
                o, m = c["observed"], c["predicted"]
                ratio = es["ratios"][fit][f"X {q}"][f"placement {p}"]
                row[fit] = {
                    "observed": o["value"],
                    "observed_se": o["se"],
                    "predicted": m["value"],
                    "predicted_se": m["se"],
                    "observed_minus_predicted": o["value"] - m["value"],
                    "O_over_M": ratio["ratio"],
                    "O_over_M_ci95": ratio["ci95"],
                }
                if q == "lambda":
                    row[fit]["systematic (fit window)"] = o["systematic (fit window)"]
            if q == "lambda":
                gap = {fit: row[fit]["observed_minus_predicted"] for fit in FITS}
                row["share of the log-linear gap removed by the binomial fit"] = (
                    1 - gap["binomial"] / gap["wls"]
                )
                row["binomial minus log-linear, observed / predicted"] = [
                    row["binomial"]["observed"] - row["wls"]["observed"],
                    row["binomial"]["predicted"] - row["wls"]["predicted"],
                ]
            blk[q] = row
        out[f"pair {p}"] = blk
    out["all ten (O/M)"] = {
        q: {
            fit: {
                "ratio": es["ratios"][fit][q]["all ten"]["ratio"],
                "ci95": es["ratios"][fit][q]["all ten"]["ci95_factor_at_least_1"],
            }
            for fit in FITS
        }
        for q in X_TESTS
    }
    tests: dict = {}
    for fit in FITS:
        tests[fit] = {}
        for label in ("prespecified", "idle_noise_only"):
            tests[fit][label] = {
                q: {
                    k: v[k] for k in ("chi2", "p", "abs_z_below_2", "p_holm_over_the_three_X_tests")
                }
                for q, v in pt["chi2"][fit][label].items()
            }
    out["X tests [chi2, p, |z| < 2, Holm p] by fit and sigma_f"] = tests
    out["pool, prespecified rule on the binomial fit"] = pt["pool"]["after the run"][
        "binomial fit, all ten (same rule)"
    ]
    # the table's values are the effect-size table's: check one cell against the raw fits
    lam = [D.tests["binomial"][i]["X lambda"]["observed"] for i in np.flatnonzero(D.mask(1))]
    assert abs(np.mean(lam) - out["pair 1"]["lambda"]["binomial"]["observed"]) < 1e-12
    return out


# ------------------------------------------------------------------------------ rule size
def pair_day_correlation(t: dict, paired: bool) -> np.ndarray:
    """Correlation of (z_Lambda, z3, z5) for one pair-day under the model's null hypothesis.

    e = O - M per test. Observation: Lambda = eps3/eps5, so its error is Lambda (a3/O3 - a5/O5)
    to first order; likewise the simulator's. f: one value for a paired pair-day, whose three
    stored f terms (signed) then move together; two independent ones for a split pair-day,
    whose Lambda term is the two relative terms in quadrature (fits.ratio_test).
    """
    L, e3, e5 = t["X lambda"], t["X eps3"], t["X eps5"]
    rel = {k: t[k]["observed"] for k in X_TESTS}
    pred = {k: t[k]["predicted"] for k in X_TESTS}
    var = {
        k: t[k]["observed_uncertainty"] ** 2 + t[k]["simulator_noise"] ** 2 + t[k]["from_f"] ** 2
        for k in X_TESTS
    }
    obs_part = {
        k: rel["X lambda"] * t[k]["observed_uncertainty"] ** 2 / rel[k]
        for k in ("X eps3", "X eps5")
    }
    sim_part = {
        k: pred["X lambda"] * t[k]["simulator_noise"] ** 2 / pred[k] for k in ("X eps3", "X eps5")
    }
    if paired:
        f_part = {k: L["from_f"] * t[k]["from_f"] for k in ("X eps3", "X eps5")}
        f35 = e3["from_f"] * e5["from_f"]
    else:
        f_part = {k: pred["X lambda"] * t[k]["from_f"] ** 2 / pred[k] for k in ("X eps3", "X eps5")}
        f35 = 0.0
    if not paired:  # Lambda's split term falls as eps5's prediction rises
        f_part["X eps5"] = -f_part["X eps5"]
    c3 = obs_part["X eps3"] + sim_part["X eps3"] + f_part["X eps3"]
    c5 = -obs_part["X eps5"] - sim_part["X eps5"] + f_part["X eps5"]
    C = np.array(
        [
            [var["X lambda"], c3, c5],
            [c3, var["X eps3"], f35],
            [c5, f35, var["X eps5"]],
        ]
    )
    s = np.sqrt(np.diag(C))
    return C / np.outer(s, s)


def simulate_rule(corr: list[np.ndarray], rng: np.random.Generator) -> dict:
    """chi2 of each test over the pair-days, z of each pair-day drawn with its correlation
    (a matrix square root, since z_Lambda = (z3 - z5)/sqrt(2) is singular)."""
    chol = []
    for c in corr:
        w, u = np.linalg.eigh(c)
        chol.append(u * np.sqrt(np.clip(w, 0, None)))
    hits = np.zeros(3)
    any_hit = 0
    worst = []
    done = 0
    while done < REPS:
        m = min(200_000, REPS - done)
        chi = np.zeros((m, 3))
        for L in chol:
            z = rng.standard_normal((m, 3)) @ L.T
            chi += z**2
        hits += (chi > BOUND).sum(0)
        any_hit += int((chi > BOUND).any(1).sum())
        worst.append(chi.max(1))
        done += m
    worst = np.concatenate(worst)
    size = any_hit / REPS
    return {
        "each test (Lambda, eps3, eps5)": floats(hits / REPS),
        "any of the three": size,
        "mc_se": sqrt(size * (1 - size) / REPS),
        "bound for a familywise 5% (95th percentile of the largest chi2)": float(
            np.quantile(worst, 0.95)
        ),
    }


def rule_size(D: ES.Data) -> dict:
    rng = np.random.default_rng(SEED)
    simple = np.array([[1.0, sqrt(0.5), -sqrt(0.5)], [sqrt(0.5), 1.0, 0.0], [-sqrt(0.5), 0.0, 1.0]])
    own = [pair_day_correlation(t, D.pairing[i] == "paired") for i, t in enumerate(D.tests["wls"])]
    for c in own:
        assert np.linalg.eigvalsh(c).min() > 0
    pt = result("prespecified_tests.json")["chi2"]["wls"]["prespecified"]
    observed = {q: pt[q]["chi2"] for q in X_TESTS}
    out: dict = {
        "what": f"P(any of the three chi2 over the ten pair-days > {BOUND:.2f}) when the model is "
        f"right; {REPS:,} simulated experiments per case, seed {SEED}",
        "independent tests (exact)": 1 - 0.95**3,
        "z_Lambda = (z3 - z5)/sqrt(2), z3 and z5 independent": simulate_rule([simple] * 10, rng),
        "each pair-day's own correlations (WL, prespecified sigma_f)": {
            **simulate_rule(own, rng),
            "corr(z_Lambda, z3) range": [min(c[0, 1] for c in own), max(c[0, 1] for c in own)],
            "corr(z_Lambda, z5) range": [min(c[0, 2] for c in own), max(c[0, 2] for c in own)],
            "corr(z3, z5) range": [min(c[1, 2] for c in own), max(c[1, 2] for c in own)],
        },
        "bonferroni bound (chi2_10 at 1 - 0.05/3)": float(stats.chi2.ppf(1 - 0.05 / 3, 10)),
        "observed chi2 (prespecified)": observed,
        "holm p (prespecified)": {q: pt[q]["p_holm_over_the_three_X_tests"] for q in X_TESTS},
    }
    bound = out["each pair-day's own correlations (WL, prespecified sigma_f)"][
        "bound for a familywise 5% (95th percentile of the largest chi2)"
    ]
    out["rejected at the familywise-5% bound"] = {q: v > bound for q, v in observed.items()}
    return out


# ------------------------------------------------------------------------------ fit window
def fit_window() -> dict:
    w = result("prespecified_tests.json")["windows"]
    out: dict = {"what": w["what"]}
    for p in ("pair 1", "pair 2", "all ten"):
        blk = w[p]
        out[p] = {
            x: {
                "rounds 3-8 left out": blk[x]["systematic (half range, rounds 3-8 left out)"],
                "with rounds 3-8": blk[x]["half range with rounds 3-8"],
            }
            for x in ("device", "model", "device_minus_model")
        }
        out[p]["rounds 3-8 (device, model)"] = {
            k: [v["device"], v["model"]] for k, v in blk["by_choice"].items() if k.endswith("3-8")
        }
    return out


# --------------------------------------------------------------------------- lambda offset
def lambda_offset(D: ES.Data) -> dict:
    swing = result("day_swing.json")["days"]["2026-10-05"]["variants"]
    lower = {
        "q4 q5 q24 q25 at their medians": -swing["q4 q5 q24 q25"]["lambda_change"],
        "the distance-3 patch at its medians": -swing["d3"]["lambda_change"],
    }
    pt = result("prespecified_tests.json")["per_pair_day"]
    stored = result("effect_sizes.json")["ratios"]
    day2 = int(np.flatnonzero((D.day == 2) & (D.placement == 1))[0])
    day1 = int(np.flatnonzero((D.day == 1) & (D.placement == 1))[0])

    def summary(y, v) -> dict:
        r = ES.random_effects(y, v)
        return {
            "ratio": r["ratio"],
            "ci95 (floored, as the paper)": r["ci95_factor_at_least_1"],
            "ci95 (Hartung-Knapp)": r["ci95"],
            "tau": r["tau"],
            "k": r["k"],
        }

    out: dict = {
        "what": "observed over predicted Lambda_X over the pair-days: DerSimonian-Laird mean of "
        "ln(O/M), variances (sigma_O/O)^2 + (sigma_M/M)^2 (effect_sizes.random_effects)",
        "day-2 prediction lowered by (day_swing.json)": lower,
    }
    for fit in FITS:
        s = D.terms(fit, "X lambda")
        y = np.log(s.O / s.M)
        v = (s.sO / s.O) ** 2 + (s.sM / s.M) ** 2
        keep = np.arange(len(y)) != day2
        both = keep & (np.arange(len(y)) != day1)
        blk = {
            "all ten": summary(y, v),
            "without day 2 of pair 1": summary(y[keep], v[keep]),
            "without days 1 and 2 of pair 1": summary(y[both], v[both]),
        }
        assert abs(blk["all ten"]["ratio"] - stored[fit]["X lambda"]["all ten"]["ratio"]) < 1e-12
        for label, d in lower.items():
            M = s.M.copy()
            M[day2] -= d
            blk[f"day 2 lowered: {label}"] = summary(
                np.log(s.O / M), (s.sO / s.O) ** 2 + (s.sM / M) ** 2
            )
        z = np.array([r[fit]["X lambda"]["z"] for r in pt])
        blk["t test on z (prespecified sigma_f)"] = {
            "all ten": t_test(z),
            "without day 2 of pair 1": t_test(z[keep]),
        }
        out[fit] = blk
    return out


def t_test(z) -> dict:
    t = stats.ttest_1samp(z, 0.0)
    return {"mean_z": float(np.mean(z)), "t": float(t.statistic), "p": float(t.pvalue), "n": len(z)}


# ------------------------------------------------------------------------------- tracking
def tracking(D: ES.Data) -> dict:
    stored = result("effect_sizes.json")["tracking"]
    out: dict = {
        "what": "pair 1, Lambda_X: O_i = c M_i with one c, sigma_i^2 = sigma_O,i^2 + c^2 "
        "sigma_M,i^2 (effect_sizes.track); at c = 1 the chi2 is the sum of the prespecified z^2",
    }
    for fit in FITS:
        s = D.terms(fit, "X lambda")
        m = D.mask(1)
        O, M, sO, sM, day = s.O[m], s.M[m], s.sO[m], s.sM[m], D.day[m]
        blk = {}
        for label, keep in (("all days", np.ones(len(O), bool)), ("without day 2", day != 2)):
            t = ES.track(O[keep], M[keep], sO[keep], sM[keep])["follows_prediction"]
            at1 = float((((O - M) ** 2) / (sO**2 + sM**2))[keep].sum())
            dof = int(keep.sum())
            blk[label] = {
                "c": t["c"],
                "chi2 at c": t["chi2"],
                "dof at c": t["dof"],
                "p at c": t["p"],
                "chi2 at c = 1": at1,
                "dof at c = 1": dof,
                "p at c = 1": float(stats.chi2.sf(at1, dof)),
                "c = 1 against c: Delta chi2": at1 - t["chi2"],
                "c = 1 against c: p (1 dof)": float(stats.chi2.sf(at1 - t["chi2"], 1)),
            }
        ref = stored[fit]["leave_one_day_out"]["without day 2"]["follows_prediction"]
        assert abs(blk["without day 2"]["c"] - ref["c"]) < 1e-9
        out[fit] = blk
    return out


# --------------------------------------------------------------------------- detector bar
def expected_max(n: int) -> dict:
    """The largest of n independent standard normal values: mean, median, 95th percentile."""

    def density(x: float) -> float:
        return x * n * stats.norm.pdf(x) * stats.norm.cdf(x) ** (n - 1)

    mean, _ = integrate.quad(density, -10, 12, limit=200, points=[sqrt(2 * log(n))])
    return {
        "mean": float(mean),
        "median": float(stats.norm.ppf(0.5 ** (1 / n))),
        "95th percentile": float(stats.norm.ppf(0.95 ** (1 / n))),
    }


def excess(job, shots: int) -> np.ndarray:
    """Each memory detector's excess over its prediction in standard errors, as
    run._worst_detectors computes it: against the predictions from the calibrations before and
    after the job, the smaller of the two."""
    a = job.analysis
    observed = a["observed"]["settings"]
    wants = [a["predicted_at_measured_dephasing"]["settings"]]
    if a.get("predicted_from_calibration_after"):
        wants.append(a["predicted_from_calibration_after"]["settings"])
    noise = 1 / shots + 1 / R.SIMULATED_SHOTS
    out = []
    for i, s in enumerate(SETTINGS):
        if s.kind != "memory":
            continue
        got = np.array(observed[i]["detector_rates"])
        z = []
        for w in wants:
            want = np.array(w[i]["detector_rates"])
            z.append((got - want) / np.sqrt(np.maximum(want * (1 - want), 1e-6) * noise))
        out.append(np.min(z, axis=0))
    return np.concatenate(out)


def detector_bar() -> dict:
    shots = result("fits_counts.json")["jobs"]
    per_job = {}
    for job in jobs():
        ex = excess(job, int(shots[job.folder.name]["shots"]))
        n = len(ex)
        bar = sqrt(2 * log(n))
        assert abs(bar - job.analysis["chance_excess"]) < 1e-12
        stored = job.analysis["detectors_beyond_chance"]
        assert int((ex > bar).sum()) == stored
        m = expected_max(n)
        per_job[job.name] = {
            "detectors": n,
            "bar sqrt(2 ln N)": bar,
            "expected largest": m["mean"],
            "median largest": m["median"],
            "95th percentile of the largest": m["95th percentile"],
            "beyond sqrt(2 ln N)": stored,
            "beyond the expected largest": int((ex > m["mean"]).sum()),
            "beyond the 95th percentile": int((ex > m["95th percentile"]).sum()),
        }

    def span(k: str) -> list:
        return [min(v[k] for v in per_job.values()), max(v[k] for v in per_job.values())]

    def share(k: str) -> list:
        s = [v[k] / v["detectors"] for v in per_job.values()]
        return [min(s), max(s)]

    return {
        "what": "one-sided: a detector counts when it fires more often than predicted (against "
        "the predictions from the calibrations before and after the job, the smaller excess)",
        "detectors range": span("detectors"),
        "bar range": span("bar sqrt(2 ln N)"),
        "expected largest range": span("expected largest"),
        "95th percentile range": span("95th percentile of the largest"),
        "share beyond sqrt(2 ln N)": share("beyond sqrt(2 ln N)"),
        "share beyond the expected largest": share("beyond the expected largest"),
        "share beyond the 95th percentile": share("beyond the 95th percentile"),
        "per_job": per_job,
    }


# ------------------------------------------------------------------------------ f read-off
def f_read_off() -> dict:
    f = result("model_vs_device.json")["f"]
    per_job = {
        name: {
            "idle test f (median)": v["median"],
            "its error": v["median_uncertainty"],
            "from eps": v["from_memory"],
            "from eps agrees": v["from_memory_agrees"],
            "from detection rates": v["from_detector_rates"],
            "from detection rates agrees": v["from_detector_rates_agree"],
        }
        for name, v in f["per_job"].items()
    }
    return {
        "what": "run._dephasing_from_memory, fixed before the run: f read off the X-memory runs "
        "by the curve through the predictions at f = 0, the idle test's f and f = 1, from eps "
        "(distance 3 where the job has it) and from the stabilizer detection rates; agrees if "
        "|f_mem - f_idle| < 2 sqrt(sigma_mem^2 + sigma_idle^2 + 0.1^2), sigma_mem = 0 for the rates",
        "jobs": len(per_job),
        "from eps agrees": f["from_memory_agrees_in"],
        "from detection rates agrees": f["from_detector_rates_agree_in"],
        "from eps median": f["from_memory_median"],
        "from detection rates median, range": [
            f["from_detector_rates_median"],
            f["from_detector_rates_range"],
        ],
        "per_job": per_job,
    }


def main() -> None:
    D = ES.Data()
    res = {
        "binomial_primary": binomial_primary(D),
        "rule_size": rule_size(D),
        "fit_window": fit_window(),
        "lambda_offset": lambda_offset(D),
        "tracking": tracking(D),
        "detector_bar": detector_bar(),
        "f_read_off": f_read_off(),
    }
    write_json("stat_checks.json", res)
    report(res)


def report(res: dict) -> None:
    b = res["binomial_primary"]
    print(
        "Lambda_X binomial / log-linear: "
        + "; ".join(
            f"pair {p} {b[f'pair {p}']['lambda']['binomial']['observed']:.3f} / "
            f"{b[f'pair {p}']['lambda']['wls']['observed']:.3f}"
            for p in (1, 2)
        )
    )
    s = res["rule_size"]["each pair-day's own correlations (WL, prespecified sigma_f)"]
    bound = s["bound for a familywise 5% (95th percentile of the largest chi2)"]
    print(f"size of the rule: {s['any of the three']:.3f} (familywise-5% bound {bound:.2f})")
    o = res["lambda_offset"]
    print(
        "Lambda_X O/M without day 2 of pair 1: "
        + ", ".join(f"{fit} {o[fit]['without day 2 of pair 1']['ratio']:.2f}" for fit in FITS)
    )
    t = res["tracking"]["wls"]["without day 2"]
    print(f"pair 1 without day 2: c {t['c']:.3f}, p at c = 1 {t['p at c = 1']:.3f}")


if __name__ == "__main__":
    main()
