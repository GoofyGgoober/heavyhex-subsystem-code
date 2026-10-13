"""What the idle test's f looks like, and how it depends on T2 (App. f figure, Sec. agreement).

Reads only committed inputs: runs/*/analysis.json and calibration.json, results/idle_f_inputs.json
(idle_f.py's cache of <X>, <Y> and the exact references) and results/perqubit_f_decode.json.

  decay    v(n) of three data qubits of one job (day 1, repetition 1, d = 3 patch), with the
           shot noise taken out [Eq. (r2)] and the error bar of Eq. (varlnr), against the
           model's exact <X> at f = 0 and f = 1. The qubits: the patch's shortest reported T2,
           the f closest to the job's median, and the patch's largest f. Every model curve
           carries the qubit's fitted offset alpha,

             v_f(n) = exp(alpha) x0(n)^(1 - f) x1(n)^f,

           at f = 0 and 1 (the figure's band), the qubit's own f and the job's median f.
  f_vs_t2  f against IBM's reported T2 and Gamma_phi = 1/T2 - 1/(2 T1) (T2 capped at 2 T1),
           one point per physical qubit: the median over the jobs that measured it. The 11
           data qubits the idle test could not pin ("measured": false) are left out; negative
           estimates are kept. Spearman rho with a bootstrap over physical qubits. In the model
           ln x0 - ln x1 = Gamma_phi t_DD, so f = Gamma_dd / Gamma_phi, where Gamma_dd is the
           dephasing rate beyond T1 that the idle test measures. One f for every qubit means
           Gamma_dd proportional to Gamma_phi; the slope beta of ln Gamma_dd on ln Gamma_phi
           over physical qubits says how far from that they are (beta = 1: one f fits all).
  memory   f read off the X-memory runs (analysis.json "dephasing_from_memory") against the
           idle test's median, per job.
  decode   eps per round of each patch's X memory with the chip's shots decoded at per-qubit-f
           priors minus at the median-f priors (perqubit_f_decode.py), and a two-sided sign test.

Seconds, one process.

    python docs/analysis/idle_f_t2.py      # writes results/idle_f_t2.json
"""

from __future__ import annotations

from collections import defaultdict
from itertools import pairwise
from math import sqrt

import numpy as np
from scipy.stats import binomtest, rankdata, spearmanr

from common import RESULTS, RUNS, day_number, job_folders, read, result, write_json
from idle_f import fit, variances

EXAMPLE_JOB, EXAMPLE_PATCH = "ibm_fez-2026-10-04-r1", "3"
T2_BINS = (0, 20, 50, 100, np.inf)  # us
BOOT, SEED = 10_000, 1


def gamma_phi(t1: float, t2: float) -> float:
    """IBM's pure-dephasing rate (1/us), with the model's cap T2 <= 2 T1."""
    return 1 / min(t2, 2 * t1) - 1 / (2 * t1)


def offset(v, log_x0, log_x1, var_log, f) -> float:
    """alpha of the weighted line ln v = alpha + ln x0 - f (ln x0 - ln x1), at the fitted f."""
    keep = v > 0.05
    w = 1 / var_log[keep]
    extra = log_x0[keep] - np.log(v[keep])
    full = (log_x0 - log_x1)[keep]
    return float(-(w * (extra - f * full)).sum() / w.sum())


def model_curve(alpha: float, log_x0, log_x1, f: float) -> list[float]:
    """v_f(n) = exp(alpha) x0(n)^(1 - f) x1(n)^f."""
    return np.exp(alpha + log_x0 - f * (log_x0 - log_x1)).tolist()


def decay(inputs: dict) -> dict:
    """Panel (a): three qubits of one job's d = 3 patch."""
    job = inputs[EXAMPLE_JOB]
    a = read(RUNS / EXAMPLE_JOB / "analysis.json")
    cal = read(RUNS / EXAMPLE_JOB / "calibration.json")
    median = a["idle_test"]["median"]
    chips = job["chips"][EXAMPLE_PATCH]
    shares = a["idle_test"]["per_qubit"][EXAMPLE_PATCH]
    measured = [k for k, s in enumerate(shares) if s is not None and np.isfinite(s)]

    def t2(k):
        return cal["t2_us"][str(chips[k])]

    picks = {
        "shortest reported T2 in the patch": min(measured, key=t2),
        "f closest to the job median": min(measured, key=lambda k: abs(shares[k] - median)),
        "largest f in the patch": max(measured, key=lambda k: shares[k]),
    }
    assert len(set(picks.values())) == 3, picks
    qubits = []
    for role, k in picks.items():
        q = job["qubits"][f"{EXAMPLE_PATCH}/{k}"]
        with np.errstate(divide="ignore", invalid="ignore"):
            v, old, new = variances(q)
            sigma_v = v * np.sqrt(new)  # Eq. (varlnr), for error bars only
        l0, l1 = np.array(q["log_x0"]), np.array(q["log_x1"])
        f, sigma = fit(v, l0, l1, old)  # the prespecified weights: the paper's f
        assert abs(f - shares[k]) < 1e-9, (role, f, shares[k])
        alpha = offset(v, l0, l1, old, f)
        physical = chips[k]
        qubits.append(
            {
                "role": role,
                "data": f"Q{k + 1}",
                "physical_qubit": physical,
                "t1_us": cal["t1_us"][str(physical)],
                "t2_us": t2(k),
                "gamma_phi_per_us": gamma_phi(cal["t1_us"][str(physical)], t2(k)),
                "f": f,
                "sigma_f": sigma,
                "alpha": alpha,
                "v": v.tolist(),
                "sigma_v": sigma_v.tolist(),
                "in_fit": (v > 0.05).tolist(),
                "x0": np.exp(l0).tolist(),
                "x1": np.exp(l1).tolist(),
                "curve_f0": model_curve(alpha, l0, l1, 0.0),
                "curve_f1": model_curve(alpha, l0, l1, 1.0),
                "curve_own_f": model_curve(alpha, l0, l1, f),
                "curve_job_median_f": model_curve(alpha, l0, l1, median),
            }
        )
    return {
        "job": EXAMPLE_JOB,
        "day": day_number(job["day"]),
        "rep": job["rep"],
        "patch": int(EXAMPLE_PATCH),
        "rounds": job["rounds"],
        "job_median_f": median,
        "job_median_f_uncertainty": a["idle_test"]["median_uncertainty"],
        "qubits": qubits,
    }


def qubit_jobs() -> list[dict]:
    """Every data qubit of every job: its prespecified f and that day's calibration."""
    rows = []
    for folder in job_folders():
        a = read(folder / "analysis.json")
        cal = read(folder / "calibration.json")
        for r in a["dephasing_by_qubit"]:
            q, d = r["qubit"], str(r["patch"])
            f = a["idle_test"]["per_qubit"][d][int(r["data"][1:]) - 1]  # before clipping at 0
            t1, t2 = cal["t1_us"][str(q)], cal["t2_us"][str(q)]
            assert t2 == r["t2_us"]
            rows.append(
                {
                    "job": folder.name,
                    "day": day_number(a["rep"]["day"]),
                    "physical_qubit": q,
                    "patch": int(d),
                    "data": r["data"],
                    "f": f if r["measured"] else None,
                    "sigma_f": r["uncertainty"] if r["measured"] else None,
                    "measured": r["measured"],
                    "t1_us": t1,
                    "t2_us": t2,
                    "gamma_phi_per_us": gamma_phi(t1, t2),
                }
            )
    return rows


def corrected_f(inputs: dict) -> dict:
    """f with the delta-method variance of Eq. (varlnr) (idle_f.py's "new"), per (job, physical qubit)."""
    out = {}
    for name, job in inputs.items():
        for key, q in job["qubits"].items():
            d, k = key.split("/")
            with np.errstate(divide="ignore", invalid="ignore"):
                v, _, new = variances(q)
            f, _ = fit(v, np.array(q["log_x0"]), np.array(q["log_x1"]), new)
            out[name, job["chips"][d][int(k)]] = f
    return out


def resamples(n: int) -> np.ndarray:
    """BOOT bootstrap resamples of n physical qubits, the same for every statistic."""
    return np.random.default_rng(SEED).integers(0, n, size=(BOOT, n))


def spearman_boot(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rho over the points given (physical qubits), with a percentile bootstrap over them."""
    rho = spearmanr(x, y)
    idx = resamples(len(x))
    rx, ry = rankdata(x[idx], axis=1), rankdata(y[idx], axis=1)
    rx -= rx.mean(axis=1, keepdims=True)
    ry -= ry.mean(axis=1, keepdims=True)
    boot = (rx * ry).sum(1) / np.sqrt((rx**2).sum(1) * (ry**2).sum(1))
    return {
        "rho": float(rho.statistic),
        "p_value": float(rho.pvalue),
        "ci95_bootstrap_over_qubits": np.nanpercentile(boot, [2.5, 97.5]).tolist(),
        "points": len(x),
    }


def f_vs_t2(rows: list[dict], inputs: dict) -> dict:
    used = [r for r in rows if r["measured"]]
    excluded = [r for r in rows if not r["measured"]]
    by_qubit = defaultdict(list)
    for r in used:
        by_qubit[r["physical_qubit"]].append(r)

    table = []
    for q, rs in sorted(by_qubit.items()):
        f = np.array([r["f"] for r in rs])
        g = np.array([r["gamma_phi_per_us"] for r in rs])
        table.append(
            {
                "physical_qubit": q,
                "jobs": len(rs),
                "days": len({r["day"] for r in rs}),
                "t2_us_median": float(np.median([r["t2_us"] for r in rs])),
                "t2_us_range": [min(r["t2_us"] for r in rs), max(r["t2_us"] for r in rs)],
                "gamma_phi_median": float(np.median(g)),
                "f_median": float(np.median(f)),
                "f_quartiles": np.percentile(f, [25, 75]).tolist(),
                "gamma_dd_median": float(np.median(f * g)),
            }
        )
    t2 = np.array([t["t2_us_median"] for t in table])
    gphi = np.array([t["gamma_phi_median"] for t in table])
    fq = np.array([t["f_median"] for t in table])
    gdd = np.array([t["gamma_dd_median"] for t in table])

    assert (gdd > 0).all()
    lx = np.log(gphi)

    def beta(sel: np.ndarray) -> dict:
        """OLS slope of ln Gamma_dd on ln Gamma_phi over the selected qubits, bootstrapped over them."""
        x, y = lx[sel], np.log(gdd[sel])
        idx = resamples(len(x))
        xb, yb = x[idx], y[idx]
        xb = xb - xb.mean(1, keepdims=True)
        boot = (xb * (yb - yb.mean(1, keepdims=True))).sum(1) / (xb**2).sum(1)
        return {
            "beta": float(np.polyfit(x, y, 1)[0]),
            "ci95_bootstrap_over_qubits": np.percentile(boot, [2.5, 97.5]).tolist(),
            "physical_qubits": int(sel.sum()),
        }

    bins = []
    for lo, hi in pairwise(T2_BINS):
        sel = [t for t in table if lo <= t["t2_us_median"] < hi]
        bins.append(
            {
                "t2_us": [lo, None if np.isinf(hi) else hi],
                "physical_qubits": len(sel),
                "qubits": [t["physical_qubit"] for t in sel],
                "qubit_jobs": sum(t["jobs"] for t in sel),
                "median_f_over_qubits": float(np.median([t["f_median"] for t in sel])),
                "f_range_over_qubits": [
                    min(t["f_median"] for t in sel),
                    max(t["f_median"] for t in sel),
                ],
            }
        )

    # The ratio artefact: f = Gamma_dd / Gamma_phi with IBM's Gamma_phi in the denominator.
    # (1) How much a qubit's reported ln Gamma_phi moves from day to day, next to the spread
    #     across qubits. (2) Within a qubit, whether f follows the day's T2 and whether
    #     Gamma_dd follows the day's Gamma_phi.
    per_day = defaultdict(dict)  # one calibration per day: a qubit's T1, T2 repeat across jobs
    for r in used:
        per_day[r["physical_qubit"]][r["day"]] = np.log(r["gamma_phi_per_us"])
    repeat = {q: list(d.values()) for q, d in per_day.items() if len(d) >= 2}
    within_var = float(
        sum(((np.array(v) - np.mean(v)) ** 2).sum() for v in repeat.values())
        / sum(len(v) - 1 for v in repeat.values())
    )
    days_mean = float(np.mean([len(per_day[t["physical_qubit"]]) for t in table]))
    between_var = float(np.var(lx, ddof=1))
    reliability = 1 - within_var / days_mean / between_var  # errors-in-variables attenuation
    within_f, within_dd = {}, {}
    for q, rs in by_qubit.items():
        if len({r["day"] for r in rs}) >= 4:
            g = [r["gamma_phi_per_us"] for r in rs]
            within_f[q] = float(spearmanr([r["t2_us"] for r in rs], [r["f"] for r in rs]).statistic)
            within_dd[q] = float(spearmanr(g, [r["f"] * x for r, x in zip(rs, g)]).statistic)
    small = [t for t in table if t["gamma_phi_median"] < 0.005]

    new = corrected_f(inputs)
    new_by_qubit = defaultdict(list)
    for r in rows:
        fn = new[r["job"], r["physical_qubit"]]
        if np.isfinite(fn):
            new_by_qubit[r["physical_qubit"]].append((fn, r["t2_us"]))
    new_t2 = np.array([np.median([t for _, t in v]) for v in new_by_qubit.values()])
    new_f = np.array([np.median([f for f, _ in v]) for v in new_by_qubit.values()])

    all_f = np.array([r["f"] for r in used])
    return {
        "qubit_jobs": len(rows),
        "qubit_jobs_used": len(used),
        "physical_qubits": len(table),
        "excluded_measured_false": [
            {k: r[k] for k in ("job", "physical_qubit", "t1_us", "t2_us", "gamma_phi_per_us")}
            for r in excluded
        ],
        "excluded_gamma_phi_max": max(r["gamma_phi_per_us"] for r in excluded),
        "used_gamma_phi_median": float(np.median([r["gamma_phi_per_us"] for r in used])),
        "negative_f_kept": int((all_f < 0).sum()),
        "median_f_over_qubit_jobs": float(np.median(all_f)),
        "median_f_over_physical_qubits": float(np.median(fq)),
        "spearman_f_t2": spearman_boot(t2, fq),
        "spearman_f_gamma_phi": spearman_boot(gphi, fq),
        "spearman_gamma_dd_gamma_phi": spearman_boot(gphi, gdd),
        "beta_gamma_dd_on_gamma_phi": {
            "what": "OLS slope of ln Gamma_dd on ln Gamma_phi over physical qubits (per-qubit "
            "medians); 1 if one f fits every qubit, 0 if Gamma_dd ignores IBM's T2",
            **beta(np.ones(len(t2), bool)),
        },
        "leverage (the short-T2 qubits left out)": {
            f"t2_at_least_{cut}_us": {
                "spearman_f_t2": spearman_boot(t2[t2 >= cut], fq[t2 >= cut]),
                "beta_gamma_dd_on_gamma_phi": beta(t2 >= cut),
            }
            for cut in (20, 50)
        },
        "spearman_f_t2_over_qubit_jobs_not_independent": {
            "rho": float(spearmanr([r["t2_us"] for r in used], all_f).statistic),
            "points": len(used),
        },
        "spearman_f_t2_corrected_variance": {
            "rho": float(spearmanr(new_t2, new_f).statistic),
            "physical_qubits": len(new_f),
        },
        "t2_bins": bins,
        "ratio_artefact": {
            "within_qubit_var_ln_gamma_phi_day_to_day": within_var,
            "between_qubit_var_ln_gamma_phi": between_var,
            "mean_days_per_qubit": days_mean,
            "reliability_of_qubit_median_ln_gamma_phi_approx": reliability,
            "qubits_on_4_or_more_days": len(within_f),
            "within_qubit_spearman_f_t2": {
                "median": float(np.median(list(within_f.values()))),
                "positive": sum(v > 0 for v in within_f.values()),
                "by_qubit": within_f,
            },
            "within_qubit_spearman_gamma_dd_gamma_phi": {
                "median": float(np.median(list(within_dd.values()))),
                "positive": sum(v > 0 for v in within_dd.values()),
                "by_qubit": within_dd,
            },
            "qubits_gamma_phi_below_0.005": len(small),
            "their_median_f": float(np.median([t["f_median"] for t in small])),
            "others_median_f": float(
                np.median([t["f_median"] for t in table if t["gamma_phi_median"] >= 0.005])
            ),
        },
        "per_physical_qubit": table,
        "per_qubit_job": [
            {
                k: r[k]
                for k in ("job", "physical_qubit", "f", "sigma_f", "t2_us", "gamma_phi_per_us")
            }
            for r in used
        ],
    }


def memory() -> dict:
    """f from the X-memory runs against the idle test's median, per job."""
    jobs = []
    for folder in job_folders():
        a = read(folder / "analysis.json")
        m = a["dephasing_from_memory"]
        jobs.append(
            {
                "job": folder.name,
                "day": day_number(a["rep"]["day"]),
                "rep": a["rep"]["rep"],
                "from_eps": int(m["from_eps"]),
                "f_mem": m["from_eps3"],  # named eps3 in analysis.json whichever eps it used
                "sigma_mem": m["from_eps3_uncertainty"],
                "f_idle": m["idle_test"],
                "sigma_idle": m["idle_test_uncertainty"],
                "agrees_prespecified_allowance": m["eps3_agrees"],
                "at_search_edge": m["from_eps3"] >= 2.0,
                # The job's predictions were simulated at f = 0, 1 and the idle test's f; beyond
                # them f_mem is an extrapolation.
                "beyond_simulated_f": m["from_eps3"] > max(1.0, m["idle_test"]),
            }
        )
    diff = np.array([j["f_mem"] - j["f_idle"] for j in jobs])
    above = int((diff > 0).sum())
    return {
        "jobs": jobs,
        "above": above,
        "of": len(jobs),
        "sign_test_p_two_sided": binomtest(above, len(jobs)).pvalue,
        "median_f_mem": float(np.median([j["f_mem"] for j in jobs])),
        "median_f_idle": float(np.median([j["f_idle"] for j in jobs])),
        "median_difference": float(np.median(diff)),
        "difference_range": [float(diff.min()), float(diff.max())],
        "agree_within_allowance": sum(j["agrees_prespecified_allowance"] for j in jobs),
        "at_search_edge_2.0": sum(j["at_search_edge"] for j in jobs),
        "beyond_simulated_f": sum(j["beyond_simulated_f"] for j in jobs),
    }


def decode() -> dict:
    """The chip's shots decoded at per-qubit-f priors, minus at median-f priors: eps per round.

    f sets only the Z-error weights, which Z memory's decoder does not use, so Z memory must
    come out unchanged; the sign test is on X memory.
    """
    data = result("perqubit_f_decode.json")
    rows = []
    for name, job in data.items():
        assert job["median_f_reproduces_prespecified"], name
        assert job["per_qubit_f"]["Z"]["per_round"] == job["median_f"]["Z"]["per_round"], name
        a = read(RUNS / name / "analysis.json")
        for p, (median, se) in job["median_f"]["X"]["per_round"].items():
            own = job["per_qubit_f"]["X"]["per_round"][p][0]
            rows.append(
                {
                    "job": name,
                    "day": day_number(a["rep"]["day"]),
                    "rep": a["rep"]["rep"],
                    "patch": int(p),
                    "eps_median_f": median,
                    "eps_per_qubit_f": own,
                    "difference": own - median,
                    "se_eps_median_f": se,
                }
            )
    d = np.array([r["difference"] for r in rows])
    worse = int((d > 0).sum())
    return {
        "basis": "X (Z memory is unchanged in every job)",
        "patch_jobs": len(rows),
        "worse_with_per_qubit_f": worse,
        "better_with_per_qubit_f": int((d < 0).sum()),
        "sign_test_p_two_sided": binomtest(worse, int((d != 0).sum())).pvalue,
        "mean_difference": float(d.mean()),
        "mean_difference_se": float(d.std(ddof=1) / sqrt(len(d))),
        "difference_range": [float(d.min()), float(d.max())],
        "rows": rows,
    }


def main() -> None:
    inputs = read(RESULTS / "idle_f_inputs.json")
    out = {
        "what": "App. f figure and Sec. agreement: the idle-test decay against its f = 0 and f = 1 "
        "references; f against reported T2 per physical qubit; f from memory against the idle "
        "test; per-qubit-f priors in decoding. f is the prespecified per-qubit estimate.",
        "decay": decay(inputs),
        "f_vs_t2": f_vs_t2(qubit_jobs(), inputs),
        "memory": memory(),
        "decode": decode(),
    }
    write_json("idle_f_t2.json", out)

    a, s, m, x = out["decay"], out["f_vs_t2"], out["memory"], out["decode"]
    for q in a["qubits"]:
        print(
            f"q{q['physical_qubit']:<4d} T2 {q['t2_us']:6.1f} us  f {q['f']:.2f} +- {q['sigma_f']:.2f}"
            f"  alpha {q['alpha']:+.3f}  ({q['role']})"
        )
    r = s["spearman_f_t2"]
    print(
        f"rho(f, T2) over {r['points']} physical qubits {r['rho']:.2f} "
        f"[{r['ci95_bootstrap_over_qubits'][0]:.2f}, {r['ci95_bootstrap_over_qubits'][1]:.2f}]; "
        f"beta {s['beta_gamma_dd_on_gamma_phi']['beta']:.2f}"
    )
    for b in s["t2_bins"]:
        print(
            f"  T2 {b['t2_us']}: {b['physical_qubits']} qubits, {b['qubit_jobs']} qubit-jobs, "
            f"median f {b['median_f_over_qubits']:.2f}"
        )
    print(
        f"f_mem > f_idle in {m['above']}/{m['of']} (p = {m['sign_test_p_two_sided']:.1e}); "
        f"per-qubit priors worse in {x['worse_with_per_qubit_f']}/{x['patch_jobs']} "
        f"(p = {x['sign_test_p_two_sided']:.2f}), mean {100 * x['mean_difference']:+.2f} pp"
    )


if __name__ == "__main__":
    main()
