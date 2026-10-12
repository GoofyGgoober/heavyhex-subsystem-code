"""One machine-readable source for the statistics the paper reports.

Post hoc; no QPU use. Reads only files already in the repository and writes
paper_numbers.json (every number, with the file it comes from) and paper_numbers.txt
(a readable digest) beside this script. Nothing in src/, runs/ or paper/ is changed.

Sections of paper_numbers.json
  per_placement            eps3, eps5 and Lambda for X and Z memory on each placement, under the
                           prespecified weighted-log fit ("wls") and the binomial-likelihood refit
                           ("binomial"): inverse-variance mean over the five days, standard error
                           (widened by sqrt(chi2/nu) when > 1), within-placement chi2/nu, the
                           model's prediction under the same estimator, observed/predicted, and the
                           within-placement Student-t test of Lambda < 1 (nu = 4).
  placements_compared      difference between the two placements and their qubit overlap. There is
                           deliberately NO chip-level estimate or p-value (no random effects, no
                           Birge-widened two-placement mean): two overlapping placements cannot
                           support a device-wide statement.
  per_repetition           the ten repetitions under both fits (Table VI) and the per-repetition
                           verdicts.
  prespecified_pooled_record  the prespecified ten-repetition rule (record only; appendix).
  agreement_tests          the prespecified model-agreement test under both fits.
  fit_bias                 bootstrap bias of the weighted-log fit and the observed shifts.
  cross_pairing            placement-2 d3 patches paired with placement 1's d5 patch.
  drift_registered_check   the prespecified half-against-half drift check.
  xy_variance_correction   the corrected Var(ln r) of the idle test: per-qubit f, exclusions,
                           job medians, and what changes downstream (control rerun vs corrected).
  per_qubit_f_side_prediction  copied from docs/diagnosis/revision/perqubit_f/perqubit_f.json.
  calibration_scatter_check_exploratory  copied from docs/diagnosis/results/stats.json.
  heating_exploratory      copied from docs/diagnosis/revision/heating (+ heating_verify).
  decoder_audit            copied from docs/diagnosis/revision/decoder (+ decoder_verify).

Inputs (relative to the repository root): runs/combined.json, runs/*/analysis.json,
docs/diagnosis/revision/refits/refits.json, docs/diagnosis/revision/xy_fix/* (including
combined_fixed.json and combined_control.json, the combined.json that xy_fix/reanalyze.py wrote
into its scratch runs directory with and without the variance fix),
docs/diagnosis/revision/heating{,_verify}/*.json, docs/diagnosis/revision/decoder{,_verify}/*.json.
The per-qubit idle-test refit imports xy_fix/xy_fix.py, which imports the frozen analysis code
from src/ (unchanged since tag v0.2.0); its cached inputs are xy_fix/idle_inputs.json.

Run from the repository root:
    python docs/diagnosis/revision/stats/paper_numbers.py
"""

from __future__ import annotations

import glob
import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
REV = HERE.parent  # docs/diagnosis/revision

FITS = ("wls", "binomial")
BASES = ("X", "Z")
QUANTITIES = ("eps3", "eps5", "lambda")
PLACEMENTS = (1, 2)
SOURCES: set[str] = set()


def rel(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


def load(path: str | Path) -> dict:
    path = Path(path) if Path(path).is_absolute() else ROOT / path
    SOURCES.add(rel(path))
    return json.loads(path.read_text(encoding="utf-8"))


REFITS_PATH = "docs/diagnosis/revision/refits/refits.json"
REFITS = load(REFITS_PATH)
COMBINED = load("runs/combined.json")
PER_REP = REFITS["per_rep"]
KEYS = sorted(PER_REP)


def day_number(day: str) -> int:
    days = sorted({v["day"] for v in PER_REP.values()})
    return days.index(day) + 1


# ------------------------------------------------------------------------- helpers
def rep_value(entry: dict, q: str, fit: str) -> tuple[float, float]:
    """(value, sigma) of one repetition, as refits/refit.py pools it."""
    x = entry[q]
    if fit == "wls":
        return float(x["wls"][0]), float(x["wls"][1])
    if q == "lambda":
        return float(x["binomial"]), float(x["binomial_se_quasi"])
    return float(x["binomial"]), float((x["binomial_hi68_quasi"] - x["binomial_lo68_quasi"]) / 2)


def pool(values, sigmas) -> dict:
    """Inverse-variance mean; error widened by sqrt(chi2/nu) when > 1 (the prespecified rule)."""
    v, w = np.asarray(values, float), 1 / np.asarray(sigmas, float) ** 2
    m = float((w * v).sum() / w.sum())
    chi2 = float((w * (v - m) ** 2).sum())
    nu = len(v) - 1
    red = chi2 / nu if nu else 1.0
    se0 = float(1 / sqrt(w.sum()))
    return {"value": m, "se": se0 * sqrt(max(red, 1.0)), "se_unwidened": se0, "chi2": chi2,
            "nu": nu, "chi2_per_dof": red, "widened": red > 1}


def lambda_t_test(m: float, se: float, nu: int) -> dict:
    t = (m - 1) / se
    return {"t": t, "nu": nu, "one_sided_p_lambda_below_1": float(stats.t.cdf(t, df=nu))}


def verdict(lam: float, sigma: float, critical: float = 1.645) -> str:
    """run.verdict (src/heavyhex/experiment/run.py): one-sided 5% each way."""
    if lam - critical * sigma > 1:
        return "d = 5 better"
    if lam + critical * sigma < 1:
        return "d = 5 worse"
    return "undecided"


def span(values) -> list[float]:
    return [float(np.min(values)), float(np.max(values))]


def clean(x):
    """Strict JSON: numpy scalars to Python, NaN/inf to null (e.g. an excluded qubit's f)."""
    if isinstance(x, dict):
        return {k: clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if np.isfinite(x) else None
    return x


# ------------------------------------------------------------------- per placement
def per_placement() -> dict:
    out: dict = {"source": REFITS_PATH,
                 "method": "observed: inverse-variance mean over the five days, se widened by "
                           "sqrt(chi2/nu) when chi2/nu > 1 (nu = 4); binomial sigmas are half-widths "
                           "of the quasi-likelihood 68% profile intervals. predicted: plain mean over the "
                           "five days of the model's prediction (each job's measured f), fitted with the "
                           "same estimator. ratio = observed / predicted_mean; its se carries the "
                           "observed error only. Lambda is pooled from the daily ratios. The t test is "
                           "within one placement (nu = 4); there is no chip-level test."}
    checks = []
    for b in BASES:
        out[b] = {}
        for pl in PLACEMENTS:
            keys = [k for k in KEYS if PER_REP[k]["placement"] == pl]
            out[b][f"placement {pl}"] = {}
            for fit in FITS:
                block = {}
                for q in QUANTITIES:
                    vals = [rep_value(PER_REP[k][b], q, fit) for k in keys]
                    p = pool([v for v, _ in vals], [s for _, s in vals])
                    ref = REFITS["summary"]["placement_pooled"][f"{b} {q} {fit} placement {pl}"]
                    checks.append(max(abs(p["value"] - ref["pooled"]), abs(p["se"] - ref["se_widened"]),
                                      abs(p["chi2_per_dof"] - ref["chi2_per_dof"])))
                    pred = [float(PER_REP[k][b]["predicted"][q][fit]) for k in keys]
                    pm = float(np.mean(pred))
                    ratios = [v / pr for (v, _), pr in zip(vals, pred)]
                    agree = [bool(REFITS["agreement_tests"][fit][k][f"{b} {q}"]["agrees"]) for k in keys]
                    block[q] = {
                        "observed": p,
                        "predicted_mean": pm,
                        "predicted_range": span(pred),
                        "observed_over_predicted": p["value"] / pm,
                        "observed_over_predicted_se_observed_only": p["se"] / pm,
                        "per_day_observed_over_predicted_range": span(ratios),
                        "days_model_agrees": int(sum(agree)),
                        "per_day": [{"day": PER_REP[k]["day"], "value": v, "se": s, "predicted": pr,
                                     "ratio": v / pr, "model_agrees": a}
                                    for k, (v, s), pr, a in zip(keys, vals, pred, agree)],
                    }
                block["lambda"]["within_placement_test"] = lambda_t_test(
                    block["lambda"]["observed"]["value"], block["lambda"]["observed"]["se"], 4)
                out[b][f"placement {pl}"][fit] = block
    assert max(checks) < 1e-12, max(checks)
    out["replicates_refits_placement_pooled_to"] = max(checks)

    ranges = {}
    for fit in FITS:
        for b in BASES:
            r = [out[b][f"placement {pl}"][fit][q]["observed_over_predicted"]
                 for pl in PLACEMENTS for q in ("eps3", "eps5")]
            ranges[f"{b} {fit}"] = span(r)
    out["observed_over_predicted_range_eps3_eps5_both_placements"] = ranges
    return out


def placements_compared() -> dict:
    """Difference between the two placements and their overlap; no chip-level inference."""
    from heavyhex.patches.placement import FEZ_MAP, build_layout

    out: dict = {"note": "No chip-level estimate, p-value or upper bound is given: on every day these were "
                         "the only usable distance-5 sites, and they overlap. Random-effects and "
                         "Birge-widened two-placement results were dropped at the outside reviewer's "
                         "request (docs/diagnosis/revision/stats/placement_stats.json still holds them "
                         "and must not be cited)."}
    for b in BASES:
        for fit in FITS:
            l1 = PP[b]["placement 1"][fit]["lambda"]["observed"]
            l2 = PP[b]["placement 2"][fit]["lambda"]["observed"]
            diff = l1["value"] - l2["value"]
            se = sqrt(l1["se"] ** 2 + l2["se"] ** 2)
            out[f"Lambda_{b} {fit}"] = {
                "placement_1_minus_placement_2": diff, "se": se,
                "chi2_between_placements_1dof": (diff / se) ** 2,
                "plain_mean_of_the_two_descriptive_only": (l1["value"] + l2["value"]) / 2,
            }
    dev = json.loads(Path(FEZ_MAP).read_text())

    def qubits(d, spot):
        origin, direction = spot
        lay = build_layout(d, dev["coords"], dev["edges"], origin=tuple(origin), direction=tuple(direction))
        return set(lay.physical_qubits)

    spots = {(r["rep"], d): set() for r in COMBINED["reps"] for d in ("3", "5")}
    for r in COMBINED["reps"]:
        for d in ("3", "5"):
            spots[(r["rep"], d)].add(tuple(map(tuple, r["spots"][d])))
    assert len(spots[(1, "5")]) == len(spots[(2, "5")]) == len(spots[(1, "3")]) == 1
    p1 = qubits(5, next(iter(spots[(1, "5")])))
    p2 = qubits(5, next(iter(spots[(2, "5")])))
    d3_p2 = [qubits(3, s) for s in sorted(spots[(2, "3")])]
    d3_p1 = qubits(3, next(iter(spots[(1, "3")])))
    out["overlap"] = {
        "source": "runs/combined.json spots + src/heavyhex/patches/placement.py (read only)",
        "d5_qubits_each": len(p1), "d5_shared": len(p1 & p2),
        "placement_2_d3_spots": [list(map(list, s)) for s in sorted(spots[(2, "3")])],
        "placement_2_d3_inside_placement_1_d5": all(q <= p1 for q in d3_p2),
        "placement_1_d3_qubits_in_placement_2_d5": len(d3_p1 & p2),
    }
    return out


# ------------------------------------------------------------------ per repetition
def per_repetition() -> dict:
    rows = []
    for k in KEYS:
        e = PER_REP[k]
        row = {"day": e["day"], "day_number": day_number(e["day"]), "rep": e["rep"],
               "placement": e["placement"], "pairing": e["pairing"]}
        for b in BASES:
            for fit in FITS:
                row[f"{b} {fit}"] = {q: list(rep_value(e[b], q, fit)) for q in QUANTITIES}
                lam, s = rep_value(e[b], "lambda", fit)
                row[f"{b} {fit}"]["verdict"] = verdict(lam, s)
            row[f"{b} predicted wls"] = {q: float(e[b]["predicted"][q]["wls"]) for q in QUANTITIES}
        rows.append(row)
    lx = {fit: [r[f"X {fit}"]["lambda"][0] for r in rows] for fit in FITS}
    reg = [r["verdict"] for r in COMBINED["reps"]]
    diff = np.array(lx["binomial"]) - np.array(lx["wls"])
    return {
        "source": REFITS_PATH + " (per_rep); registered verdicts from runs/combined.json",
        "rows": rows,
        "Lambda_X_range": {fit: span(lx[fit]) for fit in FITS},
        "registered_verdicts": {v: reg.count(v) for v in set(reg)},
        "verdicts": {f"{b} {fit}": {v: [r[f"{b} {fit}"]["verdict"] for r in rows].count(v)
                                    for v in {r[f"{b} {fit}"]["verdict"] for r in rows}}
                     for b in BASES for fit in FITS},
        "binomial_lowers_Lambda_X_in": int((diff < 0).sum()),
        "binomial_raises_Lambda_X_max": float(diff.max()),
        "eps5_binomial_minus_wls_pp": {
            b: {"mean": float(100 * np.mean([r[f"{b} binomial"]["eps5"][0] - r[f"{b} wls"]["eps5"][0] for r in rows])),
                "max": float(100 * np.max([r[f"{b} binomial"]["eps5"][0] - r[f"{b} wls"]["eps5"][0] for r in rows]))}
            for b in BASES},
        "per_rep_observed_over_predicted_range": {
            f"{b} {q} {fit}": span([rep_value(PER_REP[k][b], q, fit)[0] / PER_REP[k][b]["predicted"][q][fit]
                                    for k in KEYS])
            for b in BASES for q in QUANTITIES for fit in FITS},
    }


# --------------------------------------------------------------- prespecified record
def prespecified_record() -> dict:
    px = COMBINED["pooled_x"]
    nu = len(COMBINED["reps"]) - 1
    record = {
        "source": "runs/combined.json (pooled_x), registered code at v0.2.0",
        "Lambda_X": px["lambda"], "se": px["lambda_uncertainty"],
        "chi2_per_dof": px["spread"], "nu": nu, "chi2": px["spread"] * nu,
        "birge_ratio": sqrt(px["spread"]), "t": (px["lambda"] - 1) / px["lambda_uncertainty"],
        "t9_critical": px["critical"], "verdict": px["verdict"],
        "predicted_inverse_variance_weighted": px["predicted"],
        "note": "Record only (appendix): the rule treats the ten repetitions as independent.",
    }
    rp = REFITS["summary"]["pooled_lambda"]
    assert abs(rp["X wls"]["lambda"] - px["lambda"]) < 1e-12
    other = {}
    for b in BASES:
        for fit in FITS:
            v = rp[f"{b} {fit}"]
            other[f"{b} {fit}"] = {
                "Lambda": v["lambda"], "se": v["se_widened"], "sqrt_chi2_per_dof": v["sqrt_chi2_per_dof"],
                "sigmas_below_one": v["sigmas_below_one"], "verdict": v["verdict"],
                "predicted_inverse_variance_weighted": v["predicted_weighted"],
                "prespecified": b == "X" and fit == "wls",
            }
    return {"registered": record, "same_rule_other_fits_and_Z": other,
            "same_rule_other_fits_source": REFITS_PATH + " (summary.pooled_lambda; placement_as_unit omitted)"}


def agreement_tests() -> dict:
    out = {"source": REFITS_PATH + " (agreement, agreement_tests); wls replicates runs/combined.json model"}
    for fit in FITS:
        model = REFITS["agreement"][fit]
        tests = REFITS["agreement_tests"][fit]
        out[fit] = {q: {k: model[q][k] for k in ("chi2", "bar", "agrees", "reps_agreeing")} for q in model}
        out[fit]["reps_agreeing_on_all_three_X"] = int(sum(
            all(tests[k][f"X {q}"]["agrees"] for q in QUANTITIES) for k in KEYS))
    for q, v in COMBINED["model"].items():
        assert abs(v["chi2"] - out["wls"][q]["chi2"]) < 1e-9
    return out


def fit_bias() -> dict:
    curves = REFITS["curves"]
    out = {"source": REFITS_PATH + " (curves[*].bootstrap: parametric, 2000 replicates; per_rep)"}
    for b in BASES:
        for d in ("3", "5"):
            cs = [c for name, c in curves.items() if name.endswith(f"{b} d{d}")]
            shift = [PER_REP[k][b][f"eps{d}"]["binomial"] - PER_REP[k][b][f"eps{d}"]["wls"][0] for k in KEYS]
            out[f"{b} eps{d}"] = {
                "wls_bias_pp_mean": 100 * float(np.mean([c["bootstrap"]["wls_bias"] for c in cs])),
                "wls_bias_pp_range": [100 * v for v in span([c["bootstrap"]["wls_bias"] for c in cs])],
                "binomial_bias_pp_mean": 100 * float(np.mean([c["bootstrap"]["binomial_bias"] for c in cs])),
                "observed_shift_binomial_minus_wls_pp_mean": 100 * float(np.mean(shift)),
            }
    gof = REFITS["goodness_of_fit"]
    out["binomial_goodness_of_fit"] = {"deviance": gof["deviance_sum"], "dof": gof["dof_sum"],
                                       "p": gof["p_total"], "curves_with_A_above_1": len(gof["A_above_1"]),
                                       "points_with_P_at_or_above_half": sum(gof["curves_with_points_at_or_above_half"].values())}
    return out


# ------------------------------------------------------- cross pairing, drift (prespecified)
def cross_pairing() -> dict:
    days = sorted({v["day"] for v in PER_REP.values()})
    out = {"source": REFITS_PATH}
    for fit in FITS:
        rows = []
        for day in days:
            r1, r2 = PER_REP[f"{day} r1"], PER_REP[f"{day} r2"]
            e3_r2, e5_r1 = rep_value(r2["X"], "eps3", fit)[0], rep_value(r1["X"], "eps5", fit)[0]
            e5_r2, e3_r1 = rep_value(r2["X"], "eps5", fit)[0], rep_value(r1["X"], "eps3", fit)[0]
            rows.append({"day": day, "rep2_eps3_over_rep1_eps5": e3_r2 / e5_r1,
                         "rep2_own": e3_r2 / e5_r2, "rep1_own": e3_r1 / e5_r1})
        out[fit] = {"per_day": rows, **{f"mean_{k}": float(np.mean([r[k] for r in rows]))
                                        for k in ("rep2_eps3_over_rep1_eps5", "rep2_own", "rep1_own")}}
    return out


def drift() -> dict:
    rel_se, changes, later_worse = [], [], 0
    for path in sorted(glob.glob(str(ROOT / "runs/*/analysis.json"))):
        for d in load(path)["drift"].values():
            early, late, change = d["early"], d["late"], d["change"]
            se = (late - early) / change
            rel_se.append(se / ((early + late) / 2))
            changes.append(change)
            later_worse += late > early
    rel_se = np.array(rel_se)
    return {
        "source": "runs/*/analysis.json (drift)",
        "comparisons": len(rel_se),
        "se_over_mean_of_halves_min": float(rel_se.min()),
        "se_over_mean_of_halves_max": float(rel_se.max()),
        "se_over_mean_of_halves_median": float(np.median(rel_se)),
        "threshold_3.5_times_median": float(3.5 * np.median(rel_se)),
        "largest_abs_change_in_se": float(np.max(np.abs(changes))),
        "prespecified_threshold_se": 3.5,
        "second_half_worse": int(later_worse),
    }


# ------------------------------------------------------------- XY variance correction
def xy_variance_correction() -> dict:
    sys.path.insert(0, str(REV / "xy_fix"))
    import xy_fix as XY  # noqa: E402  (puts src/ on the path; frozen analysis code)
    from scipy.stats import spearmanr

    SOURCES.add(rel(REV / "xy_fix/idle_inputs.json"))
    data = XY.stage1()
    ratios, per, jobs = [], [], {}
    for name, job in data.items():
        so, uo, sn, un = [], [], [], []
        for key, q in job["qubits"].items():
            d, k = key.split("/")
            with np.errstate(divide="ignore", invalid="ignore"):  # r = 0 points are cut below
                r, old, new = XY.variances(q)
            keep = r > 0.05
            ratios += list(np.sqrt(new[keep] / old[keep]))
            l0, l1 = np.array(q["log_x0"]), np.array(q["log_x1"])
            f_o, s_o = XY.fit(r, l0, l1, old)
            f_n, s_n = XY.fit(r, l0, l1, new)
            so.append(f_o)
            uo.append(s_o)
            sn.append(f_n)
            un.append(s_n)
            per.append({"job": name, "patch": d, "index": int(k), "physical_qubit": job["chips"][d][int(k)],
                        "f_old": f_o, "sigma_old": s_o, "f_new": f_n, "sigma_new": s_n,
                        "points_kept": int(keep.sum()), "smallest_r_kept": float(r[keep].min()) if keep.any() else None,
                        "_args": (r, l0, l1, new)})
        jobs[name] = {"old": XY.summary(so, uo), "new": XY.summary(sn, un),
                      "day": job["day"], "rep": job["rep"]}
    ref = load(REV / "xy_fix/xy_fix.json")
    for name, v in jobs.items():
        for ver in ("old", "new"):
            for k2, x in ref[name][ver].items():
                assert abs(v[ver][k2] - x) < 1e-12, (name, ver, k2)

    fo = np.array([p["f_old"] for p in per]); fn = np.array([p["f_new"] for p in per])
    so = np.array([p["sigma_old"] for p in per]); sn = np.array([p["sigma_new"] for p in per])
    both = np.isfinite(fo) & np.isfinite(fn)
    newly_excluded = []
    for p in per:
        if np.isfinite(p["f_old"]) and not np.isfinite(p["f_new"]):
            loosest = XY.R.LOOSEST_F
            XY.R.LOOSEST_F = float("inf")
            try:
                f_unc, s_unc = XY.fit(*p["_args"])
            finally:
                XY.R.LOOSEST_F = loosest
            newly_excluded.append({k2: v for k2, v in p.items() if k2 != "_args"} |
                                  {"f_new_uncapped": f_unc, "sigma_new_uncapped": s_unc, "cut_sigma": loosest})
    newly_included = [{k2: v for k2, v in p.items() if k2 != "_args"}
                      for p in per if not np.isfinite(p["f_old"]) and np.isfinite(p["f_new"])]
    ratios = np.array(ratios)
    df = np.abs(fn[both] - fo[both])

    def dist(f, s):
        m = np.isfinite(f)
        q1, q3 = np.percentile(f[m], [25, 75])
        return {"measured": int(m.sum()), "excluded": int((~m).sum()), "negatives": int((f[m] < 0).sum()),
                "iqr": [float(q1), float(q3)], "spearman_f_sigma": float(spearmanr(f[m], s[m]).statistic)}

    job_rows = {}
    for name, v in jobs.items():
        o, n = v["old"], v["new"]
        job_rows[name] = {"day": v["day"], "day_number": day_number(v["day"]), "rep": v["rep"],
                          "median_old": o["median"], "median_new": n["median"],
                          "change": n["median"] - o["median"],
                          "change_over_se_old": (n["median"] - o["median"]) / o["median_uncertainty"],
                          "se_old": o["median_uncertainty"], "se_new": n["median_uncertainty"],
                          "chi2_per_dof_old": o["chi2_per_dof"], "chi2_per_dof_new": n["chi2_per_dof"],
                          "weighted_mean_old": o["weighted_mean"], "weighted_mean_new": n["weighted_mean"],
                          "qubits_old": o["n"], "qubits_new": n["n"]}
    largest = max(job_rows, key=lambda j: abs(job_rows[j]["change"]))
    others = [abs(r["change"]) for j, r in job_rows.items() if j != largest]
    others_z = [abs(r["change_over_se_old"]) for j, r in job_rows.items() if j != largest]

    def ranges(ver):
        meds = [r[f"median_{ver}"] for r in job_rows.values()]
        by_day = {}
        for r in job_rows.values():
            by_day.setdefault(r["day_number"], []).append(r[f"median_{ver}"])
        return {"job_median_range": span(meds), "median_of_job_medians": float(np.median(meds)),
                "chi2_per_dof_range": span([r[f"chi2_per_dof_{ver}"] for r in job_rows.values()]),
                "weighted_mean_range": span([r[f"weighted_mean_{ver}"] for r in job_rows.values()]),
                "f_range_by_day (Table I)": {d: span(v) for d, v in sorted(by_day.items())}}

    per_qubit = {
        "points_kept_r_above_0.05": len(ratios),
        "se_ln_r_new_over_old": {"median": float(np.median(ratios)), "p95": float(np.percentile(ratios, 95)),
                                 "max": float(ratios.max()), "above_1.10": int((ratios > 1.10).sum())},
        "data_qubits": len(per),
        "old": dist(fo, so), "new": dist(fn, sn),
        "abs_change_f": {"median": float(np.median(df)), "p95": float(np.percentile(df, 95)), "max": float(df.max())},
        "abs_change_f_over_sigma_old_max": float((df / so[both]).max()),
        "sigma_f_new_over_old": {"median": float(np.median(sn[both] / so[both])),
                                 "p95": float(np.percentile(sn[both] / so[both], 95)),
                                 "max": float((sn[both] / so[both]).max())},
        "newly_excluded": newly_excluded, "newly_included": newly_included,
    }

    # downstream: the corrected rerun against an unpatched rerun in the same environment
    cmp = load(REV / "xy_fix/compare.json")
    fixed_all = load(REV / "xy_fix/reanalysis.json")
    ctrl_all = load(REV / "xy_fix/reanalysis_control.json")
    comb_fix = load(REV / "xy_fix/combined_fixed.json")
    comb_ctl = load(REV / "xy_fix/combined_control.json")
    for comb, ra in ((comb_fix, fixed_all), (comb_ctl, ctrl_all)):
        assert abs(comb["pooled_x"]["lambda"] - ra["pooled_x"]["lambda"][1]) < 1e-12
        assert all(abs(comb["model"][t]["chi2"] - ra["model"][t][1]) < 1e-9 for t in comb["model"])
    flips = []
    for a, b in zip(ctrl_all["reps"], fixed_all["reps"]):
        for t in a["tests"]:
            if a["tests"][t]["agrees"][1] != b["tests"][t]["agrees"][1]:
                flips.append({"day": a["day"], "rep": a["rep"], "test": t,
                              "control": a["tests"][t]["agrees"][1], "fixed": b["tests"][t]["agrees"][1]})
    pq_flips = []
    for rep, zc in cmp["control"]["per_qubit_f"]["z"].items():
        for t, z in zc.items():
            zf = cmp["fixed"]["per_qubit_f"]["z"][rep][t]
            if (abs(z) < 2) != (abs(zf) < 2):
                pq_flips.append({"rep": rep, "test": t, "z_control": z, "z_fixed": zf})

    def placements(comb):
        res = {}
        for t in ("X lambda", "X eps3", "X eps5", "Z lambda"):
            for pl in PLACEMENTS:
                rows = [r["tests"][t] for r in comb["reps"] if r["rep"] == pl]
                p = pool([r["observed"] for r in rows], [r["observed_uncertainty"] for r in rows])
                pm = float(np.mean([r["predicted"] for r in rows]))
                res[f"{t} placement {pl}"] = {"observed": p["value"], "se": p["se"],
                                              "chi2_per_dof": p["chi2_per_dof"], "predicted_mean": pm,
                                              "observed_over_predicted": p["value"] / pm}
        return res

    committed_obs = [r["tests"]["X lambda"]["observed"] for r in COMBINED["reps"]]
    control_obs = [r["tests"]["X lambda"]["observed"] for r in comb_ctl["reps"]]
    largest_obs = max(zip(COMBINED["reps"], comb_fix["reps"]),
                      key=lambda ab: abs(ab[1]["tests"]["X lambda"]["observed"] - ab[0]["tests"]["X lambda"]["observed"]))
    downstream = {
        "source": "docs/diagnosis/revision/xy_fix/{compare,reanalysis,reanalysis_control,combined_fixed,"
                  "combined_control}.json; control = unpatched rerun in the same environment",
        "control_observed_equals_committed_max_diff": float(np.max(np.abs(np.array(committed_obs) - control_obs))),
        "memory_f_agreements": {v: sum(bool(j["memory_f_agrees"]) for j in cmp[v]["jobs"].values())
                                for v in ("control", "fixed")},
        "median_f": {v: {k2: cmp[v]["median_f"][k2] for k2 in ("chi2", "reps_agreeing", "all_three_X", "pred_lambda_mean")}
                     for v in ("control", "fixed")},
        "per_qubit_f": {v: {k2: cmp[v]["per_qubit_f"][k2] for k2 in ("chi2", "reps_agreeing", "all_three_X", "pred_lambda_mean")}
                        for v in ("control", "fixed")},
        "pooled_x": {"committed": COMBINED["pooled_x"], "control": comb_ctl["pooled_x"], "fixed": comb_fix["pooled_x"]},
        "per_rep_verdicts_fixed": {v: [r["verdict"] for r in comb_fix["reps"]].count(v)
                                   for v in {r["verdict"] for r in comb_fix["reps"]}},
        "median_f_per_rep_agreement_flips": flips,
        "per_qubit_f_per_rep_agreement_flips_X": pq_flips,
        "per_placement_control": placements(comb_ctl),
        "per_placement_fixed": placements(comb_fix),
        "largest_observed_Lambda_X_change": {
            "day": largest_obs[0]["day"], "rep": largest_obs[0]["rep"],
            "committed": largest_obs[0]["tests"]["X lambda"]["observed"],
            "fixed": largest_obs[1]["tests"]["X lambda"]["observed"],
            "note": "observed values move because the decoder weights depend on f",
        },
        "largest_predicted_eps3_change": max(
            ({"day": a["day"], "rep": a["rep"], "control": a["tests"]["X eps3"]["predicted"],
              "fixed": b["tests"]["X eps3"]["predicted"]}
             for a, b in zip(comb_ctl["reps"], comb_fix["reps"])),
            key=lambda r: abs(r["fixed"] - r["control"])),
    }
    model_fail_both = all(not comb_ctl["model"][t]["agrees"] and not comb_fix["model"][t]["agrees"]
                          for t in comb_fix["model"])
    downstream["verdict_changes"] = {
        "pooled_verdict_changed": comb_ctl["pooled_x"]["verdict"] != comb_fix["pooled_x"]["verdict"],
        "any_per_rep_Lambda_verdict_changed": [r["verdict"] for r in comb_ctl["reps"]] != [r["verdict"] for r in comb_fix["reps"]],
        "any_model_agreement_decision_changed": not model_fail_both,
        "median_f_reps_agreeing_changed": cmp["control"]["median_f"]["reps_agreeing"] != cmp["fixed"]["median_f"]["reps_agreeing"],
        "per_qubit_f_reps_agreeing_changed": cmp["control"]["per_qubit_f"]["reps_agreeing"] != cmp["fixed"]["per_qubit_f"]["reps_agreeing"],
    }
    return {
        "source": "docs/diagnosis/revision/xy_fix/xy_fix.py (functions reused here), idle_inputs.json, xy_fix.json",
        "what": "old: Var(ln r) = (1 - r^2)/(N r^2) as registered (run.py:1059); new: delta method for "
                "separately sampled <X> and <Y>, Var(ln r) = [x^2(1-x^2)/N_X + y^2(1-y^2)/N_Y]/r^4 "
                "(with the (1+1/N) shot-noise factors).",
        "per_qubit": per_qubit,
        "jobs": job_rows,
        "largest_job_change": {"job": largest, **job_rows[largest]},
        "other_jobs_max_abs_change": max(others), "other_jobs_max_abs_change_over_se": max(others_z),
        "old": ranges("old"), "new": ranges("new"),
        "downstream": downstream,
    }


# ---------------------------------------- side analyses quoted in the paper (copied)
def per_qubit_f_side() -> dict:
    src = "docs/diagnosis/revision/perqubit_f/perqubit_f.json"
    r = load(src)
    model = r["versions"]["per_qubit_f"]["model"]
    reps = r["versions"]["per_qubit_f"]["reps"]
    pl = r["placements"]
    return {
        "source": src + " (versions.per_qubit_f; committed analysis.json predictions, 50,000 shots)",
        "chi2": {q: model[q]["chi2"] for q in model}, "bar": model["X lambda"]["bar"],
        "agrees": {q: model[q]["agrees"] for q in model},
        "reps_agreeing": {q: model[q]["reps_agreeing"] for q in model},
        "reps_agreeing_on_all_three_X": int(sum(all(e["tests"][f"X {q}"]["agrees"] for q in QUANTITIES) for e in reps)),
        "predicted_Lambda_X_mean": {f"placement {p}": {"median_f": pl["median_f"][str(p)]["X lambda"]["predicted_mean"],
                                                       "per_qubit_f": pl["per_qubit_f"][str(p)]["X lambda"]["predicted_mean"]}
                                    for p in PLACEMENTS},
    }


def calibration_scatter_check() -> dict:
    src = "docs/diagnosis/results/stats.json"
    a = load(src)["agreement_with_scatter"]
    return {"source": src + " (agreement_with_scatter; docs/diagnosis/stats.py item 4)",
            "label": "exploratory; the added spread comes from the predictions only (calibrations and idle-test f)",
            **a}


# ------------------------------------------------------------------------ heating
def heating() -> dict:
    hp = "docs/diagnosis/revision/heating/heating.json"
    h = load(hp)
    v = load("docs/diagnosis/revision/heating_verify/v_heating.json")
    x = load("docs/diagnosis/revision/heating_verify/v_extra.json")
    o = load("docs/diagnosis/revision/heating_verify/v_order.json")

    def metric(key, vkey=None):
        a = h["time_H1"][key]["across_jobs"]
        m = {"random_effects_DL": {"change": a["random"]["slope"], "se": a["random"]["se"],
                                   "ci95": a["random"]["ci95"], "tau": a["random"]["tau"]},
             "fixed_effect": {"change": a["fixed"]["slope"], "se": a["fixed"]["se"]},
             "plain_t_over_jobs_ci95": a["plain_mean_t"]["ci95"],
             "jobs": a["jobs"], "positive_slopes": a["positive_slopes"],
             "z_min": a["z_min"], "z_max": a["z_max"], "stouffer_z": a["stouffer_z"],
             "jobs_rising_bonferroni": a["jobs_rising_bonferroni"],
             "jobs_falling_bonferroni": a["jobs_falling_bonferroni"]}
        if vkey in x["hksj"]:
            m["hartung_knapp_ci95 (verify)"] = x["hksj"][vkey]["ci"]
        return m

    det = metric("detection_events", "detection_events")
    eps = h["time_H1"]["logical_eps_per_round"]["across_jobs"]
    anc = h["time_H1"]["idle_ancilla_per_qubit"]
    vq = x["per_qubit_idle_anc"]
    step = h["ordering"]["step"]
    clock = h["job_wall_clock"]
    return {
        "source": hp + "; verified in docs/diagnosis/revision/heating_verify/{v_heating,v_extra,v_order}.json",
        "what": h["what"],
        "bins_per_job": h["bins"], "bonferroni_z": h["bonferroni_z"],
        "qpu_seconds_billed_per_job": sorted({c[2] for c in clock.values()}),
        "shot_seconds_per_job_ibm_estimate": span(list(h["seconds_per_job_of_shots"].values())),
        "detection_events": det,
        "detection_events_rise_excluded_above (95% upper ends)": {
            "DL": det["random_effects_DL"]["ci95"][1], "HKSJ": det["hartung_knapp_ci95 (verify)"][1],
            "plain_t": det["plain_t_over_jobs_ci95"][1]},
        "detection_events_single_fall_not_robust (verify)": {
            "job": "2026-10-04-r2-d3", **{k: v["detection_events"]["per_job"]["2026-10-04-r2-d3"][k]
                                           for k in ("slope", "se", "z", "curve_se")},
            "z_with_curve_se": v["detection_events"]["per_job"]["2026-10-04-r2-d3"]["slope"]
            / v["detection_events"]["per_job"]["2026-10-04-r2-d3"]["curve_se"]},
        "detection_events_d3": metric("detection_events_d3", "det_d3"),
        "detection_events_d5": metric("detection_events_d5", "det_d5"),
        "flags_relays_Z_memory": metric("flags_relays_Z_memory", "flags_Z"),
        "logical_failure": metric("logical_failure", "fail"),
        "logical_eps_per_round_per_bin": {"random_effects_DL": eps["random"], "positive_jobs": eps["positive_slopes"],
                                          "series_positive (verify)": [v["eps"]["series_positive"], v["eps"]["series"]],
                                          "verify_own_weighting": v["eps"]["dl"]},
        "idle_ancilla_ones": metric("idle_ancilla_ones", "idle_anc"),
        "idle_data_ones_Y_readout": metric("idle_data_ones_Y_readout", "idle_data_Y"),
        "idle_data_ones_X_readout": metric("idle_data_ones_X_readout"),
        "idle_contrast_loss": {"random_effects_DL": h["time_H1"]["idle_contrast_loss"]["across_jobs"]["random"],
                               "verify_bootstrap_se": v["contrast"]["dl"]},
        "idle_phase_drift": h["time_H1"]["idle_phase_drift"],
        "idle_ancillas_per_qubit": {k: anc[k] for k in ("qubit_jobs", "positive", "z_above_3", "z_below_minus_3",
                                                         "expected_beyond_3_each_side_if_no_drift", "median_slope")},
        "idle_ancillas_per_qubit_verify": {k: vq[k] for k in ("n", "positive", "z_gt3", "z_lt_m3", "expected_each",
                                                              "median_slope")},
        "step": {k: step[k] for k in ("job", "qubit", "switch_shot_index", "rate_before", "rate_after",
                                      "pieces", "pieces_lower_after", "sign_test_p")},
        "step_verify": {k: o["A"][k] for k in ("pooled_changepoint", "pieces_lower_after_pooled_cp",
                                               "per_piece_changepoint_median", "block1_minus_block0_mean_rel")},
        "adjacent_shot_correlation_lag1": {"interleaved": h["ordering"]["adjacent_shot_correlation"]["H1 lag 1"],
                                           "piece_by_piece": h["ordering"]["adjacent_shot_correlation"]["H2 lag 1"]},
        "pieces_covary": {k: h["ordering"]["common_mode"][k] for k in
                          ("mean_pair_corr_same_job", "null_mean", "null_sd", "p_one_sided", "mean_pair_corr_cross_job")},
        "alternative_piece_order_detection_events": h["piece_order_H2"]["detection_events"]["pooled"],
        "registered_drift_reproduced_max_abs_difference": h["reproduce_registered_drift"]["max_abs_difference"],
    }


# ------------------------------------------------------------------------ decoder
def decoder() -> dict:
    sp = "docs/diagnosis/revision/decoder/summary.json"
    s = load(sp)
    vs = load("docs/diagnosis/revision/decoder_verify/verify_summary.json")
    rd = load("docs/diagnosis/revision/decoder/redecode.json")
    fs = s["fault_scan"]
    vf = vs["fault_scan"]
    hw = s["hardware"]
    sim = s["model_simulation redecode_sim20000"]["pooled"]
    vsim = vs["model_sim_20000_pooled"]
    mean_eps = hw["mean_eps_over_reps (mean, se over reps)"]
    shifts = {k: 100 * (mean_eps[f"uniform {k}"][0] - mean_eps[f"calibrated {k}"][0])
              for k in ("X eps3", "X eps5", "Z eps3", "Z eps5")}
    uni_x = [(r["day"], r["rep"], *r["uniform"]["X lambda"]) for r in rd["reps"]]
    worst = max(uni_x, key=lambda r: r[2])
    ties = vs["uniform_tie_check_12seeds"]
    keys = ("circuits", "mechanisms", "undetectable_single", "single_miscorrections_calibrated",
            "circuits_with_single_miscorrection_calibrated", "single_miscorrections_uniform")
    pkeys = ("pairs", "undetectable_logical_pairs", "pair_miscorrections_calibrated", "pair_miscorrections_uniform")
    return {
        "source": sp + "; redecode.json; verified in docs/diagnosis/revision/decoder_verify/verify_summary.json",
        "circuit_distance (exact<=4 [None = >4], stim graphlike, stim search)":
            s["distances_seen (exact<=4 [None = >4], stim graphlike, stim search)"],
        "fault_scan": {b: {k: fs[b][k] for k in keys + (pkeys if b.endswith("d5") else ())} for b in
                       ("X d3", "X d5", "Z d3", "Z d5")},
        "fault_scan_agrees_with_verify": all(
            vf[b]["single_mis_cal"] == fs[b]["single_miscorrections_calibrated"]
            and vf[b]["single_mis_uni"] == fs[b]["single_miscorrections_uniform"] for b in vf),
        "X d5 single miscorrections jobs": vf["X d5"]["jobs_single_mis_cal"],
        "uniform_ties_12_seeds (verify)": {
            "Z d3 singles": {"wrong_with_pymatching_tiebreak": ties["Z d3 singles rounds (1, 2, 3, 4, 6, 8)"]["mis_uniform"],
                             "wrong_under_every_perturbed_tiebreak": ties["Z d3 singles rounds (1, 2, 3, 4, 6, 8)"]["mis_all_jitter"]},
            "Z d5 pairs, rounds 1-2": {"wrong_with_pymatching_tiebreak": ties["Z d5 pairs rounds (1, 2)"]["mis_uniform"],
                                       "wrong_under_every_perturbed_tiebreak (upper bound)": ties["Z d5 pairs rounds (1, 2)"]["mis_all_jitter"]}},
        "matching_graph": {
            "shortest_logical_edges": {b: s["matching_graph"][b]["decoder_graph_hops_seen"] for b in s["matching_graph"]},
            "lightest_logical_weight_range": {b: s["matching_graph"][b]["weighted_distance_W_range"] for b in s["matching_graph"]},
            "edge_probability_range (verify)": [min(vf[b]["edge_p_min"] for b in vf if "edge_p_min" in vf[b]),
                                                max(vf[b]["edge_p_max_range"][1] for b in vf)],
            "fraction_edges_heavier_than_half_lightest_logical": {
                b: s["matching_graph"][b]["fraction_edges_heavier_than_half_W"] for b in s["matching_graph"]},
            "z_d5_flag_relay_pieces (verify)": vs["remnants_and_boundary_feed"]["Z"]["n"],
        },
        "counterexample_X_d3 (verify)": vs["counterexamples"]["cases"][0] | {"folder": vs["counterexamples"]["folder"]},
        "hardware": {
            "pooled_prespecified_rule": hw["pooled"],
            "only_one_decoder_fails": hw["mcnemar"],
            "mean_eps_over_reps": mean_eps,
            "mean_eps_shift_uniform_minus_calibrated_pp": shifts,
            "mean_eps_shift_range_pp": span(list(shifts.values())),
            "per_rep_Lambda_X_uniform_all_below_1": all(r[2] < 1 for r in uni_x),
            "per_rep_Lambda_X_uniform_verdicts": [verdict(r[2], r[3]) for r in uni_x],
            "per_rep_Lambda_X_uniform_largest": {"day": worst[0], "rep": worst[1], "value": worst[2], "se": worst[3]},
            "X_difference_bootstrap_verify": {k: vs["hardware"]["pooled"]["X diff"][k] for k in
                                              ("uni_minus_cal", "boot_sd_fixed_weights", "boot_sd_reweighted", "boot_mean_fixed_weights")},
            "points_dropped_P_ge_half (verify)": vs["hardware_points_dropped_P_ge_half"],
        },
        "model_simulation_20000_shots": {
            "main": {k: sim[k]["lambda"] for k in sim},
            "verify": {k: vsim[k]["lambda"] for k in vsim},
            "Lambda_X_drop_uniform": [sim["calibrated X"]["lambda"] - sim["uniform X"]["lambda"],
                                      vsim["X cal"]["lambda"] - vsim["X uni"]["lambda"]],
        },
    }


# ---------------------------------------------------------------------------- main
def fmt_pm(v, s, d=3):
    return f"{v:.{d}f} +- {s:.{d}f}"


def digest(res: dict) -> str:
    L = []
    pp = res["per_placement"]
    L.append("1. Per placement (inverse-variance mean over 5 days; chi2/nu with nu = 4; ratio = observed / mean predicted)")
    for b in BASES:
        for pl in PLACEMENTS:
            for fit in FITS:
                blk = pp[b][f"placement {pl}"][fit]
                parts = []
                for q in QUANTITIES:
                    o = blk[q]["observed"]
                    scale, d = (100, 1) if q != "lambda" else (1, 3)
                    parts.append(f"{q} {fmt_pm(scale * o['value'], scale * o['se'], d)} (chi2/nu {o['chi2_per_dof']:.2f}; "
                                 f"pred {scale * blk[q]['predicted_mean']:.{d}f}; ratio {blk[q]['observed_over_predicted']:.3f}; "
                                 f"agrees {blk[q]['days_model_agrees']}/5)")
                t = blk["lambda"]["within_placement_test"]
                L.append(f"   {b} placement {pl} {fit:8s}: " + "; ".join(parts) + f"; within-placement p {t['one_sided_p_lambda_below_1']:.1e}")
    for k, v in pp["observed_over_predicted_range_eps3_eps5_both_placements"].items():
        L.append(f"   observed/predicted eps3, eps5 range, {k}: {v[0]:.2f}-{v[1]:.2f}")
    pc = res["placements_compared"]
    L.append("2. Placements compared (no chip-level inference)")
    for b in BASES:
        for fit in FITS:
            v = pc[f"Lambda_{b} {fit}"]
            L.append(f"   Lambda_{b} {fit}: P1 - P2 = {fmt_pm(v['placement_1_minus_placement_2'], v['se'])}, "
                     f"chi2 (1 dof) {v['chi2_between_placements_1dof']:.1f}")
    ov = pc["overlap"]
    L.append(f"   d5 patches share {ov['d5_shared']} of {ov['d5_qubits_each']} qubits; placement-2 d3 inside "
             f"placement-1 d5: {ov['placement_2_d3_inside_placement_1_d5']}")
    pr = res["per_repetition"]
    L.append("3. Per repetition")
    L.append(f"   Lambda_X range wls {pr['Lambda_X_range']['wls'][0]:.3f}-{pr['Lambda_X_range']['wls'][1]:.3f}, "
             f"binomial {pr['Lambda_X_range']['binomial'][0]:.3f}-{pr['Lambda_X_range']['binomial'][1]:.3f}; "
             f"verdicts {pr['verdicts']}")
    L.append(f"   binomial lowers Lambda_X in {pr['binomial_lowers_Lambda_X_in']} of 10 (max rise {pr['binomial_raises_Lambda_X_max']:.4f}); "
             f"eps5 shift (pp) {pr['eps5_binomial_minus_wls_pp']}")
    for k in ("X eps3 wls", "X eps3 binomial", "Z eps3 wls", "Z eps5 wls", "Z eps5 binomial"):
        v = pr["per_rep_observed_over_predicted_range"][k]
        L.append(f"   per-rep observed/predicted {k}: {v[0]:.2f}-{v[1]:.2f}")
    rec = res["prespecified_pooled_record"]["registered"]
    L.append("4. Prespecified 10-repetition rule (record only)")
    L.append(f"   Lambda_X {fmt_pm(rec['Lambda_X'], rec['se'])}, chi2/nu {rec['chi2_per_dof']:.2f} (nu 9), Birge "
             f"{rec['birge_ratio']:.2f}, t {rec['t']:.1f} vs {rec['t9_critical']:.3f}: {rec['verdict']}; predicted {rec['predicted_inverse_variance_weighted']:.3f}")
    for k, v in res["prespecified_pooled_record"]["same_rule_other_fits_and_Z"].items():
        L.append(f"   same rule, {k}: {fmt_pm(v['Lambda'], v['se'])} ({v['verdict']})")
    L.append("5. Agreement tests (chi2 vs 18.31; reps agreeing)")
    for fit in FITS:
        a = res["agreement_tests"][fit]
        L.append(f"   {fit}: " + ", ".join(f"{q} {a[q]['chi2']:.1f} ({a[q]['reps_agreeing']})" for q in
                                         ("X lambda", "X eps3", "X eps5", "Z lambda", "Z eps3", "Z eps5"))
                 + f"; all three X: {a['reps_agreeing_on_all_three_X']}")
    fb = res["fit_bias"]
    L.append("6. Weighted-log bias (bootstrap, pp): " + ", ".join(
        f"{k} {fb[k]['wls_bias_pp_mean']:+.2f} (shift {fb[k]['observed_shift_binomial_minus_wls_pp_mean']:+.2f})"
        for k in ("X eps3", "X eps5", "Z eps3", "Z eps5")))
    c = res["cross_pairing"]
    L.append(f"7. Cross pairing (mean): wls {c['wls']['mean_rep2_eps3_over_rep1_eps5']:.3f} vs own "
             f"{c['wls']['mean_rep2_own']:.3f}; binomial {c['binomial']['mean_rep2_eps3_over_rep1_eps5']:.3f} vs own "
             f"{c['binomial']['mean_rep2_own']:.3f}")
    d = res["drift_registered_check"]
    L.append(f"8. Registered drift check: {d['comparisons']} comparisons, max |change| {d['largest_abs_change_in_se']:.2f} SE, "
             f"SE/mean {100 * d['se_over_mean_of_halves_min']:.1f}-{100 * d['se_over_mean_of_halves_max']:.1f}% "
             f"(median {100 * d['se_over_mean_of_halves_median']:.1f}%, 3.5x = {100 * d['threshold_3.5_times_median']:.0f}%), "
             f"second half worse {d['second_half_worse']}/{d['comparisons']}")
    xy = res["xy_variance_correction"]
    q = xy["per_qubit"]
    L.append("9. XY-variance correction")
    L.append(f"   excluded {q['old']['excluded']} -> {q['new']['excluded']} of {q['data_qubits']}; newly excluded "
             + "; ".join(f"{e['job']} d{e['patch']} q{e['physical_qubit']} f {e['f_old']:.3f}, sigma {e['sigma_old']:.3f} -> "
                         f"{e['sigma_new_uncapped']:.3f} (cut {e['cut_sigma']})" for e in q["newly_excluded"]))
    L.append(f"   |df| median {q['abs_change_f']['median']:.4f}, max {q['abs_change_f']['max']:.4f} "
             f"(<= {q['abs_change_f_over_sigma_old_max']:.2f} sigma); sigma_f new/old median {q['sigma_f_new_over_old']['median']:.3f}, "
             f"max {q['sigma_f_new_over_old']['max']:.3f}; negatives {q['old']['negatives']}->{q['new']['negatives']}; "
             f"IQR {q['new']['iqr'][0]:.2f}-{q['new']['iqr'][1]:.2f}; Spearman {q['old']['spearman_f_sigma']:.3f}->{q['new']['spearman_f_sigma']:.3f}")
    lj = xy["largest_job_change"]
    L.append(f"   largest job change {lj['job']}: median f {lj['median_old']:.3f} -> {lj['median_new']:.3f} "
             f"({lj['change_over_se_old']:+.2f} sigma, qubits {lj['qubits_old']}->{lj['qubits_new']}); other jobs <= "
             f"{xy['other_jobs_max_abs_change']:.3f} (<= {xy['other_jobs_max_abs_change_over_se']:.2f} sigma)")
    for ver in ("old", "new"):
        r = xy[ver]
        L.append(f"   {ver}: job medians {r['job_median_range'][0]:.2f}-{r['job_median_range'][1]:.2f} (median "
                 f"{r['median_of_job_medians']:.2f}); chi2/nu {r['chi2_per_dof_range'][0]:.0f}-{r['chi2_per_dof_range'][1]:.0f}; "
                 f"weighted mean {r['weighted_mean_range'][0]:.2f}-{r['weighted_mean_range'][1]:.2f}; by day "
                 + ", ".join(f"{dd}: {v[0]:.2f}-{v[1]:.2f}" for dd, v in r["f_range_by_day (Table I)"].items()))
    ds = xy["downstream"]
    for ver in ("median_f", "per_qubit_f"):
        c0, c1 = ds[ver]["control"], ds[ver]["fixed"]
        L.append(f"   {ver}: " + ", ".join(f"{t} {c0['chi2'][t]:.1f}->{c1['chi2'][t]:.1f} ({c0['reps_agreeing'][t]}->{c1['reps_agreeing'][t]})"
                                         for t in c0["chi2"]) + f"; all three X {c0['all_three_X']}->{c1['all_three_X']}")
    L.append(f"   pooled X control {fmt_pm(ds['pooled_x']['control']['lambda'], ds['pooled_x']['control']['lambda_uncertainty'])} -> "
             f"fixed {fmt_pm(ds['pooled_x']['fixed']['lambda'], ds['pooled_x']['fixed']['lambda_uncertainty'])}; verdict changes {ds['verdict_changes']}")
    L.append(f"   per-rep agreement flips (median f): {ds['median_f_per_rep_agreement_flips']}; per-qubit f (X): {ds['per_qubit_f_per_rep_agreement_flips_X']}")
    for pl in PLACEMENTS:
        a, b = ds["per_placement_control"][f"X lambda placement {pl}"], ds["per_placement_fixed"][f"X lambda placement {pl}"]
        L.append(f"   Lambda_X placement {pl}: {fmt_pm(a['observed'], a['se'])} -> {fmt_pm(b['observed'], b['se'])}; "
                 f"predicted {a['predicted_mean']:.3f} -> {b['predicted_mean']:.3f}")
    pq = res["per_qubit_f_side_prediction"]
    L.append("9b. Per-qubit-f side prediction (committed): chi2 " + ", ".join(f"{q} {pq['chi2'][q]:.1f}" for q in ("X lambda", "X eps3", "X eps5"))
             + "; predicted Lambda_X " + ", ".join(f"{k} {v['median_f']:.3f}->{v['per_qubit_f']:.3f}" for k, v in pq["predicted_Lambda_X_mean"].items()))
    cs = res["calibration_scatter_check_exploratory"]
    L.append("9c. Calibration-scatter check (exploratory): " + ", ".join(
        f"{q} chi2 {cs[q]['chi2']:.1f}->{cs[q]['chi2_with_scatter']:.1f} (scatter {100 * cs[q]['relative_scatter']:.0f}%)"
        for q in ("X lambda", "X eps3", "X eps5")))
    h = res["heating_exploratory"]
    de = h["detection_events"]
    L.append("10. Heating (exploratory; fractional change start->end of a job)")
    L.append(f"   detection events DL {100 * de['random_effects_DL']['change']:+.2f}% "
             f"[{100 * de['random_effects_DL']['ci95'][0]:+.2f}, {100 * de['random_effects_DL']['ci95'][1]:+.2f}]; "
             f"rise excluded above (DL/HKSJ/t) " + "/".join(f"{100 * v:.2f}%" for v in h["detection_events_rise_excluded_above (95% upper ends)"].values()))
    lf = h["logical_failure"]
    L.append(f"   logical failure {100 * lf['random_effects_DL']['change']:+.2f}% [{100 * lf['random_effects_DL']['ci95'][0]:+.2f}, "
             f"{100 * lf['random_effects_DL']['ci95'][1]:+.2f}]; idle Y readout {100 * h['idle_data_ones_Y_readout']['random_effects_DL']['change']:+.2f}% "
             f"(Stouffer {h['idle_data_ones_Y_readout']['stouffer_z']:.1f}); ancillas z>3 {h['idle_ancillas_per_qubit']['z_above_3']}, "
             f"z<-3 {h['idle_ancillas_per_qubit']['z_below_minus_3']} of {h['idle_ancillas_per_qubit']['qubit_jobs']}; "
             f"step q{h['step']['qubit']} at shot {h['step']['switch_shot_index']} in {h['step']['pieces_lower_after']}/{h['step']['pieces']} pieces")
    dc = res["decoder_audit"]
    hw = dc["hardware"]["pooled_prespecified_rule"]
    L.append("11. Decoder audit")
    L.append(f"   distances {dc['circuit_distance (exact<=4 [None = >4], stim graphlike, stim search)']}")
    L.append(f"   Lambda_X calibrated {fmt_pm(hw['calibrated X']['lambda'], hw['calibrated X']['lambda_uncertainty'])}, uniform "
             f"{fmt_pm(hw['uniform X']['lambda'], hw['uniform X']['lambda_uncertainty'])}; Lambda_Z "
             f"{fmt_pm(hw['calibrated Z']['lambda'], hw['calibrated Z']['lambda_uncertainty'])} -> "
             f"{fmt_pm(hw['uniform Z']['lambda'], hw['uniform Z']['lambda_uncertainty'])}; eps shift "
             f"{dc['hardware']['mean_eps_shift_range_pp'][0]:.1f}-{dc['hardware']['mean_eps_shift_range_pp'][1]:.1f} pp; largest uniform rep "
             f"{fmt_pm(dc['hardware']['per_rep_Lambda_X_uniform_largest']['value'], dc['hardware']['per_rep_Lambda_X_uniform_largest']['se'])}")
    return "\n".join(L) + "\n"


PP: dict = {}


def main() -> None:
    global PP
    sys.path.insert(0, str(ROOT / "src"))
    PP = per_placement()
    result = {
        "about": {
            "script": rel(Path(__file__)),
            "post_hoc": True,
            "fits": {"wls": "prespecified weighted-log fit of ln[1 - 2P(n)] (Eq. fit), primary for the X-memory decision",
                     "binomial": "binomial maximum-likelihood refit of all round counts (App. stats)"},
            "chip_level_inference": "omitted by decision; see placements_compared.note",
        },
        "per_placement": PP,
        "placements_compared": placements_compared(),
        "per_repetition": per_repetition(),
        "prespecified_pooled_record": prespecified_record(),
        "agreement_tests": agreement_tests(),
        "fit_bias": fit_bias(),
        "cross_pairing": cross_pairing(),
        "drift_registered_check": drift(),
        "xy_variance_correction": xy_variance_correction(),
        "per_qubit_f_side_prediction": per_qubit_f_side(),
        "calibration_scatter_check_exploratory": calibration_scatter_check(),
        "heating_exploratory": heating(),
        "decoder_audit": decoder(),
    }
    result["about"]["inputs"] = sorted(SOURCES)
    (HERE / "paper_numbers.json").write_text(json.dumps(clean(result), indent=1, allow_nan=False) + "\n",
                                             encoding="utf-8")
    text = digest(result)
    (HERE / "paper_numbers.txt").write_text(text, encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
