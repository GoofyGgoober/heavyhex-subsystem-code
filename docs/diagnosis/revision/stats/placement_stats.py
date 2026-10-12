"""Placement-level statistics for Lambda_X and Lambda_Z (post hoc, exploratory).

Reads only committed data: runs/combined.json (the prespecified per-repetition Lambda and
its standard error) and, as a cross-check, runs/*/analysis.json (the per-job epsilon fits,
from which every repetition's Lambda and its error are recomputed with the prespecified
formula Lambda = eps3/eps5, Var(Lambda)/Lambda^2 = Var(eps3)/eps3^2 + Var(eps5)/eps5^2).

For each basis:
  (1) per placement (grouped by the distance-5 placement, i.e. by repetition number):
      inverse-variance mean over the five days, chi^2 and chi^2/nu (nu = 4), and the error
      widened by the Birge ratio sqrt(chi^2/nu) only when chi^2/nu > 1;
  (2) the two placement values (with the errors from (1)) combined by inverse variance:
      chi^2 (nu = 1), Birge ratio, widened error, t = (Lambda - 1)/se for H0: Lambda >= 1
      and its one-sided p from Student's t with nu = 1;
  (3) DerSimonian-Laird random-effects combination of the two placements (tau, mean, se,
      one-sided p with nu = 1; normal-theory p and Hartung-Knapp se given for reference);
  (4) the unweighted mean of the two placement means, se = sd/sqrt(2) = |m1 - m2|/2,
      one-sided p with nu = 1; the same on ln Lambda as a sensitivity check;
  (5) the prespecified rule (inverse variance over all 10 reps, Birge widening when
      chi^2/nu > 1, Student's t with nu = 9), reproduced and checked against
      combined.json's pooled_x.
If docs/diagnosis/revision/refits/ holds results, they are listed (see refit_summary()).

    python docs/diagnosis/revision/stats/placement_stats.py
"""

from __future__ import annotations

import json
from math import exp, sqrt
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm
from scipy.stats import t as student_t

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
RUNS = REPO / "runs"
REFITS = HERE.parent / "refits"


# ---------------------------------------------------------------- data

def load_reps() -> list[dict]:
    """Per-repetition Lambda_X, Lambda_Z (value, se, predicted) from combined.json."""
    c = json.loads((RUNS / "combined.json").read_text())
    reps = []
    for r in c["reps"]:
        row = {"day": r["day"], "rep": r["rep"], "pairing": r["pairing"], "folders": r["folders"],
               "d5_spot": r["spots"]["5"], "d3_spot": r["spots"]["3"]}
        for b in "XZ":
            t = r["tests"][f"{b} lambda"]
            row[b] = {"lambda": t["observed"], "se": t["observed_uncertainty"], "predicted": t["predicted"]}
        reps.append(row)
    return reps, c


def recompute_from_analysis(reps: list[dict]) -> dict:
    """Recompute each rep's Lambda from the job folders' eps fits; return max deviations."""
    dev = {"X": [0.0, 0.0], "Z": [0.0, 0.0]}
    for r in reps:
        tests = {}
        for folder in r["folders"]:
            a = json.loads((RUNS / folder / "analysis.json").read_text())
            tests.update(a["agreement"]["tests"])
        for b in "XZ":
            e3, e5 = tests[f"{b} eps3"], tests[f"{b} eps5"]
            lam = e3["observed"] / e5["observed"]
            se = lam * sqrt((e3["observed_uncertainty"] / e3["observed"]) ** 2
                            + (e5["observed_uncertainty"] / e5["observed"]) ** 2)
            dev[b][0] = max(dev[b][0], abs(lam - r[b]["lambda"]))
            dev[b][1] = max(dev[b][1], abs(se - r[b]["se"]))
    return {b: {"max_abs_dev_lambda": v[0], "max_abs_dev_se": v[1]} for b, v in dev.items()}


# ---------------------------------------------------------------- statistics

def ivw(x, s, widen: bool = True) -> dict:
    """Inverse-variance mean, chi^2 about it, Birge ratio, error widened only if chi^2/nu > 1."""
    x, s = np.asarray(x, float), np.asarray(s, float)
    w = 1 / s**2
    m = float((w * x).sum() / w.sum())
    se0 = float(1 / sqrt(w.sum()))
    k = len(x)
    nu = k - 1
    chi2 = float((w * (x - m) ** 2).sum())
    red = chi2 / nu if nu > 0 else float("nan")
    birge = sqrt(red) if nu > 0 else float("nan")
    scale = max(1.0, birge) if (widen and nu > 0) else 1.0
    return {"mean": m, "se_stat": se0, "chi2": chi2, "nu": nu, "chi2_per_nu": red,
            "chi2_p": float(chi2_dist.sf(chi2, nu)) if nu > 0 else float("nan"),
            "birge": birge, "widened": bool(widen and nu > 0 and red > 1), "se": se0 * scale}


def t_test(mean: float, se: float, nu: int) -> dict:
    """H0: Lambda >= 1 against Lambda < 1: t = (mean - 1)/se, one-sided p = P(T_nu <= t)."""
    t = (mean - 1) / se
    crit = float(student_t.ppf(0.95, nu))
    return {"t": t, "nu": nu, "p_one_sided": float(student_t.cdf(t, nu)), "t_crit_95": crit,
            "upper_95_one_sided": mean + crit * se,
            "verdict": "d = 5 worse" if mean + crit * se < 1 else ("d = 5 better" if mean - crit * se > 1 else "undecided")}


def dersimonian_laird(x, s) -> dict:
    x, s = np.asarray(x, float), np.asarray(s, float)
    k = len(x)
    w = 1 / s**2
    m_fe = (w * x).sum() / w.sum()
    q = float((w * (x - m_fe) ** 2).sum())
    c = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (q - (k - 1)) / c)
    ws = 1 / (s**2 + tau2)
    m = float((ws * x).sum() / ws.sum())
    se = float(1 / sqrt(ws.sum()))
    # Hartung-Knapp(-Sidik-Jonkman) variance, for reference
    se_hk = float(sqrt((ws * (x - m) ** 2).sum() / ((k - 1) * ws.sum())))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 else 0.0
    return {"Q": q, "C": c, "tau2": tau2, "tau": sqrt(tau2), "I2": i2, "weights": (ws / ws.sum()).tolist(),
            "mean": m, "se": se, "test_t1": t_test(m, se, k - 1),
            "p_one_sided_normal": float(norm.cdf((m - 1) / se)),
            "se_hartung_knapp": se_hk, "test_hartung_knapp_t1": t_test(m, se_hk, k - 1)}


def unweighted(means) -> dict:
    m = np.asarray(means, float)
    k = len(m)
    mean = float(m.mean())
    se = float(m.std(ddof=1) / sqrt(k))
    return {"mean": mean, "se": se, "test": t_test(mean, se, k - 1)}


def unweighted_log(means) -> dict:
    lm = np.log(np.asarray(means, float))
    k = len(lm)
    mu, se = float(lm.mean()), float(lm.std(ddof=1) / sqrt(k))
    t = mu / se  # H0: ln Lambda >= 0
    crit = float(student_t.ppf(0.95, k - 1))
    return {"geo_mean": exp(mu), "ln_mean": mu, "ln_se": se, "t": t, "nu": k - 1,
            "p_one_sided": float(student_t.cdf(t, k - 1)),
            "upper_95_one_sided": exp(mu + crit * se)}


# ---------------------------------------------------------------- refits (optional)

def refit_summary() -> dict | None:
    if not REFITS.is_dir():
        return None
    files = sorted(p for p in REFITS.rglob("*.json"))
    if not files:
        return {"present": True, "json_files": []}
    return {"present": True, "json_files": [str(p.relative_to(REPO)).replace("\\", "/") for p in files]}


# ---------------------------------------------------------------- main

def analyse(reps: list[dict], basis: str) -> dict:
    out: dict = {"per_placement": {}}
    placements = {1: [r for r in reps if r["rep"] == 1], 2: [r for r in reps if r["rep"] == 2]}
    pm, ps = [], []
    for p, rows in placements.items():
        x = [r[basis]["lambda"] for r in rows]
        s = [r[basis]["se"] for r in rows]
        res = ivw(x, s)
        res["days"] = [r["day"] for r in rows]
        res["values"] = x
        res["ses"] = s
        res["predicted_mean"] = float(np.mean([r[basis]["predicted"] for r in rows]))
        res["predicted_range"] = [float(min(r[basis]["predicted"] for r in rows)),
                                  float(max(r[basis]["predicted"] for r in rows))]
        res["d5_spot"] = rows[0]["d5_spot"]
        res["d3_spots"] = sorted({json.dumps(r["d3_spot"]) for r in rows})
        res["test_nu4"] = t_test(res["mean"], res["se"], 4)
        out["per_placement"][str(p)] = res
        pm.append(res["mean"])
        ps.append(res["se"])

    fe = ivw(pm, ps)  # nu = 1
    fe["test_t1"] = t_test(fe["mean"], fe["se"], 1)
    fe["I2"] = max(0.0, (fe["chi2"] - 1) / fe["chi2"]) if fe["chi2"] > 0 else 0.0
    out["two_placements_inverse_variance"] = fe
    out["two_placements_dersimonian_laird"] = dersimonian_laird(pm, ps)
    out["two_placements_unweighted"] = unweighted(pm)
    out["two_placements_unweighted_log"] = unweighted_log(pm)

    x = [r[basis]["lambda"] for r in reps]
    s = [r[basis]["se"] for r in reps]
    pre = ivw(x, s)
    pre["test_t9"] = t_test(pre["mean"], pre["se"], len(x) - 1)
    pre["predicted_ivw"] = float(sum(r[basis]["predicted"] / r[basis]["se"] ** 2 for r in reps)
                                 / sum(1 / r[basis]["se"] ** 2 for r in reps))
    out["prespecified_10_reps"] = pre
    return out


def fmt(m, s, nd=3):
    return f"{m:.{nd}f} ± {s:.{nd}f}"


def table(res: dict) -> str:
    lines = []
    for b in "XZ":
        r = res[b]
        lines.append(f"Lambda_{b}" + ("" if b == "X" else "  (Z memory: reported, not prespecified for decision)"))
        for p in ("1", "2"):
            q = r["per_placement"][p]
            lines.append(f"  placement {p}: {fmt(q['mean'], q['se'])}   chi2/nu = {q['chi2']:.2f}/4 = {q['chi2_per_nu']:.2f}"
                         f" (p = {q['chi2_p']:.2g}){'  [Birge-widened x' + format(q['birge'], '.2f') + ']' if q['widened'] else ''}"
                         f"   predicted mean {q['predicted_mean']:.2f}")
        fe = r["two_placements_inverse_variance"]
        t = fe["test_t1"]
        lines.append(f"  (2) IVW of placements: {fmt(fe['mean'], fe['se_stat'])} (stat) -> chi2 = {fe['chi2']:.1f} (nu=1, p = {fe['chi2_p']:.2g}),"
                     f" Birge {fe['birge']:.2f} -> {fmt(fe['mean'], fe['se'])}; t = {t['t']:.2f}, p1(t_1) = {t['p_one_sided']:.3f},"
                     f" 95% upper {t['upper_95_one_sided']:.2f}, {t['verdict']}")
        dl = r["two_placements_dersimonian_laird"]
        t = dl["test_t1"]
        lines.append(f"  (3) DerSimonian-Laird: tau = {dl['tau']:.3f}, {fmt(dl['mean'], dl['se'])}; t = {t['t']:.2f}, p1(t_1) = {t['p_one_sided']:.3f},"
                     f" 95% upper {t['upper_95_one_sided']:.2f}, {t['verdict']}  [normal p = {dl['p_one_sided_normal']:.2g}; HK se {dl['se_hartung_knapp']:.3f}, p1 = {dl['test_hartung_knapp_t1']['p_one_sided']:.3f}]")
        uw = r["two_placements_unweighted"]
        t = uw["test"]
        lg = r["two_placements_unweighted_log"]
        lines.append(f"  (4) unweighted mean of placements: {fmt(uw['mean'], uw['se'])}; t = {t['t']:.2f}, p1(t_1) = {t['p_one_sided']:.3f},"
                     f" 95% upper {t['upper_95_one_sided']:.2f}, {t['verdict']}  [on ln Lambda: {lg['geo_mean']:.3f}, p1 = {lg['p_one_sided']:.3f}, upper {lg['upper_95_one_sided']:.2f}]")
        pre = r["prespecified_10_reps"]
        t = pre["test_t9"]
        lines.append(f"  (5) prespecified, 10 reps: {fmt(pre['mean'], pre['se_stat'])} (stat), chi2/nu = {pre['chi2']:.1f}/9 = {pre['chi2_per_nu']:.2f},"
                     f" Birge {pre['birge']:.2f} -> {fmt(pre['mean'], pre['se'])}; t = {t['t']:.1f}, p1(t_9) = {t['p_one_sided']:.1g},"
                     f" t_9 crit {t['t_crit_95']:.3f}, {t['verdict']}")
        lines.append("")
    return "\n".join(lines)


def latex(res: dict) -> str:
    rows = [r"\begin{tabular}{lcc}",
            r" & $\LX$ & $\LZ$ \\", r"\hline"]

    def cell(m, s):
        return f"${m:.2f}\\pm{s:.2f}$"

    def get(b, key):
        return res[b][key]
    for p in ("1", "2"):
        cx, cz = get("X", "per_placement")[p], get("Z", "per_placement")[p]
        rows.append(f"Placement {p} (5 days; $\\chi^2/\\nu$) & {cell(cx['mean'], cx['se'])} ({cx['chi2_per_nu']:.1f}) & "
                    f"{cell(cz['mean'], cz['se'])} ({cz['chi2_per_nu']:.1f}) \\\\")
    def pcell(m, s, p):
        return f"{cell(m, s)}, $p={p:.2f}$"
    fx, fz = get("X", "two_placements_inverse_variance"), get("Z", "two_placements_inverse_variance")
    rows.append(f"Inverse variance, Birge-widened ($t_1$) & {pcell(fx['mean'], fx['se'], fx['test_t1']['p_one_sided'])} & "
                f"{pcell(fz['mean'], fz['se'], fz['test_t1']['p_one_sided'])} \\\\")
    dx, dz = get("X", "two_placements_dersimonian_laird"), get("Z", "two_placements_dersimonian_laird")
    rows.append(f"Random effects, DL ($t_1$) & {pcell(dx['mean'], dx['se'], dx['test_t1']['p_one_sided'])} & "
                f"{pcell(dz['mean'], dz['se'], dz['test_t1']['p_one_sided'])} \\\\")
    ux, uz = get("X", "two_placements_unweighted"), get("Z", "two_placements_unweighted")
    rows.append(f"Unweighted mean of placements ($t_1$) & {pcell(ux['mean'], ux['se'], ux['test']['p_one_sided'])} & "
                f"{pcell(uz['mean'], uz['se'], uz['test']['p_one_sided'])} \\\\")
    px, pz = get("X", "prespecified_10_reps"), get("Z", "prespecified_10_reps")

    def sci(v):
        e = int(np.floor(np.log10(v)))
        return f"{v / 10**e:.0f}\\times10^{{{e}}}"
    rows.append(f"Prespecified, 10 reps ($t_9$) & {cell(px['mean'], px['se'])}, $p={sci(px['test_t9']['p_one_sided'])}$ & "
                f"{cell(pz['mean'], pz['se'])}, $p={sci(pz['test_t9']['p_one_sided'])}$ \\\\")
    rows.append(r"\end{tabular}")
    return "\n".join(rows) + "\n"


def main() -> None:
    reps, combined = load_reps()
    check = recompute_from_analysis(reps)
    res = {"source": "runs/combined.json tests['X lambda'/'Z lambda'] observed and observed_uncertainty",
           "cross_check_vs_analysis_json": check}
    for b in "XZ":
        res[b] = analyse(reps, b)
    reg = combined["pooled_x"]
    mine = res["X"]["prespecified_10_reps"]
    res["prespecified_matches_combined_pooled_x"] = {
        "lambda_diff": mine["mean"] - reg["lambda"], "se_diff": mine["se"] - reg["lambda_uncertainty"],
        "spread_diff": mine["chi2_per_nu"] - reg["spread"], "critical_diff": mine["test_t9"]["t_crit_95"] - reg["critical"],
        "verdict_same": mine["test_t9"]["verdict"] == reg["verdict"]}
    res["refits"] = refit_summary()
    (HERE / "placement_stats.json").write_text(json.dumps(res, indent=1) + "\n", encoding="utf-8")
    txt = table(res)
    (HERE / "placement_stats_table.txt").write_text(txt, encoding="utf-8")
    (HERE / "placement_stats_table.tex").write_text(latex(res), encoding="utf-8")
    print(json.dumps(check))
    print(json.dumps(res["prespecified_matches_combined_pooled_x"]))
    print(txt)
    print("refits:", res["refits"])


if __name__ == "__main__":
    main()
