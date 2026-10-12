"""Independent re-computation of the placement-level Lambda statistics.

Reads per-job runs/*/run.json + runs/*/analysis.json directly (not combined.json) to
rebuild each repetition's Lambda and its standard error, then recomputes:
 (1) per-placement inverse-variance mean, chi2/nu (nu=4), Birge widening if chi2/nu>1
 (2) inverse-variance combination of the two placements, chi2 (nu=1), Birge, t, p (t_1)
 (3) DerSimonian-Laird random effects (k=2), p from t_1
 (4) unweighted mean of placement means, se = |m1-m2|/2, p from t_1
 (5) prespecified rule (IV over 10 reps, Birge, t_9)
Also: closed-form checks, Monte Carlo check of DL formula, alternative choices.
Writes verify_results.json next to this file. Local data only.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[4]
RUNS = ROOT / "runs"
OUT = Path(__file__).resolve().parent


def load_jobs():
    jobs = []
    for folder in sorted(p for p in RUNS.iterdir() if p.is_dir()):
        run = json.loads((folder / "run.json").read_text())
        ana = json.loads((folder / "analysis.json").read_text())
        rep = run.get("rep")
        if not rep or run.get("offline"):
            continue
        jobs.append(dict(folder=folder.name, rep=rep, tests=ana["agreement"]["tests"],
                         ana_rep=ana.get("rep")))
    return jobs


def ratio(e3, e5):
    lam = e3["observed"] / e5["observed"]
    rel = math.hypot(e3["observed_uncertainty"] / e3["observed"],
                     e5["observed_uncertainty"] / e5["observed"])
    return lam, lam * rel


def build_reps(jobs):
    reps = {}
    for j in jobs:
        key = (j["rep"]["day"], j["rep"]["rep"])
        e = reps.setdefault(key, dict(day=key[0], rep=key[1], folders=[], spots={}, tests={}))
        e["folders"].append(j["folder"])
        e["spots"].update(j["rep"]["spots"])
        e["tests"].update(j["tests"])
        # sanity: folder name agrees with run.json rep / day
        assert key[0] in j["folder"] and f"-r{key[1]}" in j["folder"], j["folder"]
        assert j["ana_rep"] == j["rep"], j["folder"]
    out = []
    for key in sorted(reps):
        e = reps[key]
        row = dict(day=e["day"], rep=e["rep"], folders=e["folders"],
                   d5_spot=e["spots"]["5"], d3_spot=e["spots"]["3"])
        for b in "XZ":
            lam_r, se_r = ratio(e["tests"][f"{b} eps3"], e["tests"][f"{b} eps5"])
            if f"{b} lambda" in e["tests"]:  # paired job: prespecified value from analysis
                lam, se = (e["tests"][f"{b} lambda"]["observed"],
                           e["tests"][f"{b} lambda"]["observed_uncertainty"])
                row[f"{b}_source"] = "analysis.json lambda test"
            else:
                lam, se = lam_r, se_r
                row[f"{b}_source"] = "ratio of eps tests (independent)"
            row[b] = (lam, se)
            row[f"{b}_ratio_formula"] = (lam_r, se_r)
        out.append(row)
    return out


def iv(x, s):
    x, s = np.asarray(x, float), np.asarray(s, float)
    w = 1 / s**2
    m = float((w * x).sum() / w.sum())
    se0 = float(1 / math.sqrt(w.sum()))
    chi2 = float((w * (x - m) ** 2).sum())
    nu = len(x) - 1
    return m, se0, chi2, nu


def p_t(lam, se, nu):
    t = (lam - 1) / se
    return t, float(stats.t.cdf(t, nu))


def analyse(reps, b):
    res = {}
    groups = {1: [r for r in reps if r["rep"] == 1], 2: [r for r in reps if r["rep"] == 2]}
    pl = {}
    for k, g in groups.items():
        assert len(g) == 5 and len({r["day"] for r in g}) == 5
        x = [r[b][0] for r in g]
        s = [r[b][1] for r in g]
        m, se0, chi2, nu = iv(x, s)
        fac = math.sqrt(max(1.0, chi2 / nu))
        se = se0 * fac
        t, p = p_t(m, se, nu)
        pl[k] = dict(days=[r["day"] for r in g], values=x, errors=s, mean=m, se_stat=se0,
                     chi2=chi2, nu=nu, chi2_per_nu=chi2 / nu,
                     chi2_p=float(stats.chi2.sf(chi2, nu)), widen=fac, se=se, t=t, p_nu4=p,
                     d5_spots=sorted({json.dumps(r["d5_spot"]) for r in g}),
                     d3_spots=[json.dumps(r["d3_spot"]) for r in g])
    res["placement"] = pl
    m1, s1 = pl[1]["mean"], pl[1]["se"]
    m2, s2 = pl[2]["mean"], pl[2]["se"]

    # (2) IV of placements + Birge
    m, se0, chi2, nu = iv([m1, m2], [s1, s2])
    birge = math.sqrt(chi2 / nu)
    se = se0 * max(1.0, birge)
    t, p = p_t(m, se, 1)
    crit = float(stats.t.ppf(0.95, 1))
    res["iv_birge"] = dict(mean=m, se_stat=se0, chi2=chi2, chi2_p=float(stats.chi2.sf(chi2, 1)),
                           birge=birge, se=se, t=t, p1=p, upper95=m + crit * se)
    # closed form for k=2: chi2 = d^2/(v1+v2)
    res["iv_birge"]["chi2_closed_form"] = (m1 - m2) ** 2 / (s1**2 + s2**2)

    # (2') same but per-placement errors NOT widened
    m_u, se0_u, chi2_u, _ = iv([m1, m2], [pl[1]["se_stat"], pl[2]["se_stat"]])
    se_u = se0_u * max(1.0, math.sqrt(chi2_u))
    t_u, p_u = p_t(m_u, se_u, 1)
    res["iv_birge_unwidened_inputs"] = dict(mean=m_u, se=se_u, chi2=chi2_u, t=t_u, p1=p_u)

    # (3) DerSimonian-Laird
    v = np.array([s1**2, s2**2])
    x = np.array([m1, m2])
    w = 1 / v
    Q = float((w * (x - (w * x).sum() / w.sum()) ** 2).sum())
    C = float(w.sum() - (w**2).sum() / w.sum())
    tau2 = max(0.0, (Q - 1) / C)
    ws = 1 / (v + tau2)
    mdl = float((ws * x).sum() / ws.sum())
    sedl = float(1 / math.sqrt(ws.sum()))
    tdl, pdl = p_t(mdl, sedl, 1)
    # Hartung-Knapp
    q_hk = float((ws * (x - mdl) ** 2).sum() / (len(x) - 1))
    se_hk = sedl * math.sqrt(q_hk)
    t_hk, p_hk = p_t(mdl, se_hk, 1)
    res["dl"] = dict(Q=Q, tau2=tau2, tau=math.sqrt(tau2), I2=max(0.0, (Q - 1) / Q), mean=mdl,
                     se=sedl, t=tdl, p1=pdl, p_normal=float(stats.norm.cdf(tdl)),
                     upper95=mdl + crit * sedl, weights=(ws / ws.sum()).tolist(),
                     tau2_closed_form=max(0.0, ((m1 - m2) ** 2 - s1**2 - s2**2) / 2),
                     hk_se=se_hk, hk_t=t_hk, hk_p1=p_hk)

    # (4) unweighted mean of placement means
    mu = (m1 + m2) / 2
    seu = float(np.std([m1, m2], ddof=1) / math.sqrt(2))
    tu, pu = p_t(mu, seu, 1)
    res["unweighted"] = dict(mean=mu, se=seu, se_closed=abs(m1 - m2) / 2, t=tu, p1=pu,
                             upper95=mu + crit * seu)
    # (4') on ln scale
    l1, l2 = math.log(m1), math.log(m2)
    ml = (l1 + l2) / 2
    sel = abs(l1 - l2) / 2
    res["unweighted_log"] = dict(mean=math.exp(ml), t=ml / sel, p1=float(stats.t.cdf(ml / sel, 1)))

    # (5) prespecified over 10 reps
    xs = [r[b][0] for r in reps]
    ss = [r[b][1] for r in reps]
    m10, se010, chi10, nu10 = iv(xs, ss)
    se10 = se010 * math.sqrt(max(1.0, chi10 / nu10))
    t10, p10 = p_t(m10, se10, nu10)
    crit9 = float(stats.t.ppf(0.95, nu10))
    verdict = ("d = 5 worse" if m10 + crit9 * se10 < 1 else
               "d = 5 better" if m10 - crit9 * se10 > 1 else "undecided")
    res["prereg"] = dict(mean=m10, se=se10, chi2=chi10, nu=nu10, chi2_per_nu=chi10 / nu10,
                         birge=math.sqrt(chi10 / nu10), t=t10, p1=p10, crit=crit9, verdict=verdict)

    # Alternative: random-effects/one-way view with reps nested in placements (variance
    # components via simple ANOVA on unweighted values) - for context only
    return res


def main():
    jobs = load_jobs()
    reps = build_reps(jobs)
    comb = json.loads((RUNS / "combined.json").read_text())
    # compare with combined.json
    cmp = []
    for r, c in zip(reps, comb["reps"]):
        assert (r["day"], r["rep"]) == (c["day"], c["rep"])
        for b in "XZ":
            t = c["tests"][f"{b} lambda"]
            cmp.append(dict(day=r["day"], rep=r["rep"], basis=b,
                            dlam=r[b][0] - t["observed"], dse=r[b][1] - t["observed_uncertainty"],
                            d_ratio_formula=r[b][0] - r[f"{b}_ratio_formula"][0],
                            dse_ratio_formula=r[b][1] - r[f"{b}_ratio_formula"][1]))
    out = dict(reps=[{k: v for k, v in r.items()} for r in reps], compare_combined=cmp)
    for b in "XZ":
        out[b] = analyse(reps, b)
    px = comb["pooled_x"]
    out["prereg_vs_combined"] = dict(
        dlam=out["X"]["prereg"]["mean"] - px["lambda"],
        dse=out["X"]["prereg"]["se"] - px["lambda_uncertainty"],
        dspread=out["X"]["prereg"]["chi2_per_nu"] - px["spread"],
        dcrit=out["X"]["prereg"]["crit"] - px["critical"],
        verdict_combined=px["verdict"], verdict_mine=out["X"]["prereg"]["verdict"])
    (OUT / "verify_results.json").write_text(json.dumps(out, indent=1, default=str))

    # print summary
    print("max |dLambda| vs combined:", max(abs(c["dlam"]) for c in cmp),
          " max |dse|:", max(abs(c["dse"]) for c in cmp))
    print("max |registered - ratio formula| lam/se:",
          max(abs(c["d_ratio_formula"]) for c in cmp), max(abs(c["dse_ratio_formula"]) for c in cmp))
    for r in reps:
        print(r["day"], r["rep"], r["d5_spot"], r["d3_spot"],
              "X %.4f+-%.4f (%s)" % (*r["X"], r["X_source"][:8]),
              "Z %.4f+-%.4f (%s)" % (*r["Z"], r["Z_source"][:8]))
    for b in "XZ":
        R = out[b]
        print(f"\n===== Lambda_{b}")
        for k in (1, 2):
            p = R["placement"][k]
            print(f" placement {k}: {p['mean']:.4f} stat {p['se_stat']:.4f} chi2 {p['chi2']:.3f}/4"
                  f"={p['chi2_per_nu']:.3f} (p={p['chi2_p']:.3f}) widen {p['widen']:.3f} se {p['se']:.4f}"
                  f" t {p['t']:.2f} p(nu=4) {p['p_nu4']:.2e}  d5 {p['d5_spots']} d3 {p['d3_spots']}")
        a = R["iv_birge"]
        print(f" (2) {a['mean']:.4f} stat {a['se_stat']:.4f} chi2 {a['chi2']:.3f} (closed {a['chi2_closed_form']:.3f}) "
              f"p {a['chi2_p']:.2e} birge {a['birge']:.3f} se {a['se']:.4f} t {a['t']:.3f} p1 {a['p1']:.4f} up95 {a['upper95']:.3f}")
        a = R["iv_birge_unwidened_inputs"]
        print(f" (2u) {a['mean']:.4f} se {a['se']:.4f} chi2 {a['chi2']:.2f} t {a['t']:.3f} p1 {a['p1']:.4f}")
        a = R["dl"]
        print(f" (3) tau {a['tau']:.4f} (closed {math.sqrt(a['tau2_closed_form']):.4f}) I2 {a['I2']:.3f} "
              f"{a['mean']:.4f} +- {a['se']:.4f} t {a['t']:.3f} p1 {a['p1']:.4f} pN {a['p_normal']:.2e} "
              f"up95 {a['upper95']:.3f} HK se {a['hk_se']:.4f} p1 {a['hk_p1']:.4f}")
        a = R["unweighted"]
        print(f" (4) {a['mean']:.4f} +- {a['se']:.4f} t {a['t']:.3f} p1 {a['p1']:.4f} up95 {a['upper95']:.3f}")
        a = R["unweighted_log"]
        print(f" (4log) {a['mean']:.4f} t {a['t']:.3f} p1 {a['p1']:.4f}")
        a = R["prereg"]
        print(f" (5) {a['mean']:.4f} +- {a['se']:.4f} chi2 {a['chi2']:.2f}/9={a['chi2_per_nu']:.3f} birge {a['birge']:.3f}"
              f" t {a['t']:.2f} p1 {a['p1']:.2e} crit {a['crit']:.4f} {a['verdict']}")
    print("\nprereg vs combined.json:", out["prereg_vs_combined"])


if __name__ == "__main__":
    main()
