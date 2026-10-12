"""Verifier: summary numbers from v_results.json, and comparison with ../refits/per_rep.csv."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2d
from scipy.stats import t as tdist

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
res = json.loads((HERE / "v_results.json").read_text())
combined = json.loads((REPO / "runs" / "combined.json").read_text())
out = {}

# ---- 1. WLS reimplementation vs stored
dmax = 0
for r in res:
    for key, c in r["curves"].items():
        dmax = max(dmax, abs(c["wls"]["eps"] - c["stored"][0]), abs(c["wls"]["se"] - c["stored"][1]))
print(f"max |my WLS - stored| over eps and se: {dmax:.2e}")

# ---- 2. per-rep table
print("\nPer repetition (eps in %): WLS +- se -> MLE [quasi]  ; Lambda WLS+-se -> MLE [quasi] [prof]")
for r in res:
    s = f"P{r['rep']} d{r['day_index']}: "
    for b in "XZ":
        for d in "35":
            c = r["curves"][f"{b}{d}"]
            s += f"{b}{d} {100*c['wls']['eps']:.1f}±{100*c['wls']['se']:.1f}->{100*c['mle']['eps']:.1f}[{100*c['mle']['quasi'][0]:.1f},{100*c['mle']['quasi'][1]:.1f}] "
        L = r["lam"][b]
        s += f"L{b} {L['wls']:.3f}±{L['wls_se']:.3f}->{L['mle']:.3f}[{L['quasi'][0]:.3f},{L['quasi'][1]:.3f}]p[{L['prof'][0]:.3f},{L['prof'][1]:.3f}] | "
    print(s)

# ---- 3. compare with the main analysis's per_rep.csv (../refits/)
theirs = {}
with open(REPO / "docs/diagnosis/revision/refits/per_rep.csv") as f:
    for row in csv.DictReader(f):
        theirs[(row["day"], int(row["rep"]), row["basis"], row["quantity"])] = row
diffs = {"binomial": 0, "quasi_lo68": 0, "quasi_hi68": 0, "prof_lo68": 0, "prof_hi68": 0, "wls": 0, "wls_se": 0}
worst = {}
for r in res:
    for b in "XZ":
        for q in ("eps3", "eps5", "lambda"):
            t = theirs[(r["day"], r["rep"], b, q)]
            if q == "lambda":
                L = r["lam"][b]
                mine = dict(binomial=L["mle"], quasi_lo68=L["quasi"][0], quasi_hi68=L["quasi"][1],
                            prof_lo68=L["prof"][0], prof_hi68=L["prof"][1], wls=L["wls"], wls_se=L["wls_se"])
            else:
                c = r["curves"][f"{b}{q[-1]}"]
                mine = dict(binomial=c["mle"]["eps"], quasi_lo68=c["mle"]["quasi"][0], quasi_hi68=c["mle"]["quasi"][1],
                            prof_lo68=c["mle"]["prof"][0], prof_hi68=c["mle"]["prof"][1], wls=c["wls"]["eps"], wls_se=c["wls"]["se"])
            for kk in diffs:
                d = abs(mine[kk] - float(t[kk]))
                if d > diffs[kk]:
                    diffs[kk] = d
                    worst[kk] = (r["day"], r["rep"], b, q, mine[kk], float(t[kk]))
print("\nmax abs diff vs their per_rep.csv:", {k: f"{v:.2e}" for k, v in diffs.items()})
for k, v in worst.items():
    print("  worst", k, v)

# ---- 4. means over reps
def mean_se(x):
    x = np.asarray(x)
    return x.mean(), x.std(ddof=1) / math.sqrt(len(x))

print("\nMeans over 10 reps (unweighted, ± sd/sqrt10):")
means = {}
for b in "XZ":
    for d in "35":
        for est in ("wls", "mle"):
            m, s = mean_se([r["curves"][f"{b}{d}"][est]["eps"] for r in res])
            means[f"{b} eps{d} {est}"] = (m, s)
            print(f"  {b} eps{d} {est}: {100*m:.2f} ± {100*s:.2f} %")
    for est in ("wls", "mle"):
        m, s = mean_se([r["lam"][b][est] for r in res])
        means[f"{b} lambda {est}"] = (m, s)
        print(f"  Lambda_{b} {est}: {m:.3f} ± {s:.3f}")
out["means"] = means

# ---- 5. pooled (prespecified rule)
def pool(vals, sig):
    v, s = np.asarray(vals), np.asarray(sig)
    w = 1 / s**2
    m = (w * v).sum() / w.sum()
    spread = (w * (v - m) ** 2).sum() / (len(v) - 1)
    se = math.sqrt(max(spread, 1) / w.sum())
    return m, se, spread

print("\nPooled (registered rule):")
for b in "XZ":
    for est in ("wls", "mle"):
        vals = [r["lam"][b][est] for r in res]
        sig = [r["lam"][b]["wls_se"] if est == "wls" else r["lam"][b]["sigma_q"] for r in res]
        m, se, sp = pool(vals, sig)
        crit = tdist.ppf(0.95, len(vals) - 1)
        print(f"  Lambda_{b} {est}: {m:.4f} ± {se:.4f}  chi2/nu={sp:.3f} sqrt={math.sqrt(sp):.3f}  (1-L)/se={(1-m)/se:.2f}  t9={crit:.3f}")
        out[f"pooled {b} {est}"] = (m, se, sp)
        # also with the profile (phi=1) half-width
        if est == "mle":
            sigp = [(r["lam"][b]["prof"][1] - r["lam"][b]["prof"][0]) / 2 for r in res]
            m2, se2, sp2 = pool(vals, sigp)
            print(f"     (with pure-profile half-width: {m2:.4f} ± {se2:.4f} chi2/nu={sp2:.3f})")
print("  combined.json pooled_x:", {k: combined["pooled_x"][k] for k in ("lambda", "lambda_uncertainty", "spread", "predicted")})

# ---- 6. by placement
print("\nBy placement (inverse-variance, widened by sqrt(chi2/nu) if > 1):")
for b in "XZ":
    for est in ("wls", "mle"):
        for p in (1, 2):
            rr = [r for r in res if r["rep"] == p]
            vals = [r["lam"][b][est] for r in rr]
            sig = [r["lam"][b]["wls_se"] if est == "wls" else r["lam"][b]["sigma_q"] for r in rr]
            m, se, sp = pool(vals, sig)
            out[f"placement {p} {b} lambda {est}"] = (m, se, sp)
            print(f"  Lambda_{b} {est} placement {p}: {m:.3f} ± {se:.3f} chi2/nu={sp:.2f}")
        for d in "35":
            for p in (1, 2):
                rr = [r for r in res if r["rep"] == p]
                vals = [r["curves"][f"{b}{d}"][est]["eps"] for r in rr]
                sig = [r["curves"][f"{b}{d}"]["wls"]["se"] if est == "wls" else r["curves"][f"{b}{d}"]["mle"]["sigma_q"] for r in rr]
                m, se, sp = pool(vals, sig)
                out[f"placement {p} {b} eps{d} {est}"] = (m, se, sp)
                print(f"     {b} eps{d} {est} placement {p}: {100*m:.1f} ± {100*se:.1f} (chi2/nu {sp:.2f})")

# ---- 7. placement as unit (n = 2)
print("\nPlacement as unit (n=2), X:")
for est in ("wls", "mle"):
    a, b_ = out[f"placement 1 X lambda {est}"][0], out[f"placement 2 X lambda {est}"][0]
    m = (a + b_) / 2
    se = abs(a - b_) / 2  # sd/sqrt(2) with sd = |a-b|/sqrt(2)
    t = (1 - m) / se
    p = tdist.sf(t, 1)
    print(f"  {est}: {m:.3f} ± {se:.3f}, one-sided p = {p:.3f}")
    out[f"unit2 {est}"] = (m, se, p)

# ---- 8. ln Lambda pooling and other estimators
print("\nOther estimators, X:")
for est in ("wls", "mle"):
    vals = np.array([r["lam"]["X"][est] for r in res])
    sig = np.array([r["lam"]["X"]["wls_se"] if est == "wls" else r["lam"]["X"]["sigma_q"] for r in res])
    lv, ls = np.log(vals), sig / vals
    m, se, sp = pool(lv, ls)
    e3 = np.mean([r["curves"]["X3"][est]["eps"] for r in res])
    e5 = np.mean([r["curves"]["X5"][est]["eps"] for r in res])
    print(f"  {est}: ln-pool {math.exp(m):.3f} ({math.exp(m-se):.3f}-{math.exp(m+se):.3f}) chi2/nu {sp:.2f};"
          f" unweighted mean {vals.mean():.3f}; ratio of means {e3/e5:.3f}")
    # alternative: ln sigma from asymmetric quasi interval in log space
    if est == "mle":
        lsig = np.array([(math.log(r["lam"]["X"]["quasi"][1]) - math.log(r["lam"]["X"]["quasi"][0])) / 2 for r in res])
        m2, se2, sp2 = pool(lv, lsig)
        print(f"     ln-pool with log-space quasi half-width: {math.exp(m2):.3f} ({math.exp(m2-se2):.3f}-{math.exp(m2+se2):.3f})")

# ---- 9. decisions per rep
print("\nPer-rep decisions (X): Lambda + 1.645 sigma < 1 ?")
for est in ("wls", "mle"):
    dec = []
    for r in res:
        L = r["lam"]["X"]
        if est == "wls":
            dec.append(L["wls"] + 1.645 * L["wls_se"] < 1)
        else:
            dec.append(L["mle"] + 1.645 * L["sigma_q"] < 1)
    print(f"  {est}: worse in {sum(dec)} / 10")
# Using the actual profile-likelihood 90% upper limit is a different test; also check quasi hi < 1
print("  MLE quasi 68% upper ends < 1 in", sum(r["lam"]["X"]["quasi"][1] < 1 for r in res), "/ 10")

# ---- 10. goodness of fit
dev = sum(c["mle"]["deviance"] for r in res for c in r["curves"].values())
dof = sum(c["mle"]["dof"] for r in res for c in r["curves"].values())
print(f"\nTotal deviance {dev:.1f} on {dof} dof, p = {chi2d.sf(dev, dof):.3f}")
bad = [(r["day"], r["rep"], k, round(c["mle"]["deviance"], 2), round(chi2d.sf(c["mle"]["deviance"], 4), 4)) for r in res for k, c in r["curves"].items() if chi2d.sf(c["mle"]["deviance"], 4) < 0.05]
print("curves with deviance p < 0.05:", bad)
badp = [(r["day"], r["rep"], k, round(c["mle"]["pearson"], 2), round(chi2d.sf(c["mle"]["pearson"], 4), 4)) for r in res for k, c in r["curves"].items() if chi2d.sf(c["mle"]["pearson"], 4) < 0.05]
print("curves with Pearson p < 0.05:", badp)
print("phi > 1 in", sum(c["mle"]["phi"] > 1 for r in res for c in r["curves"].values()), "of 40 curves")
Agt1 = [(r["day"], r["rep"], k, round(c["mle"]["A"], 3)) for r in res for k, c in r["curves"].items() if c["mle"]["A"] > 1]
print("A > 1:", len(Agt1), Agt1)

(HERE / "v_summary.json").write_text(json.dumps(out, indent=1))
