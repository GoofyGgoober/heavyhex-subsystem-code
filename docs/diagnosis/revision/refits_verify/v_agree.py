"""Verifier: refit the stored per-round predictions (200,000 simulated shots; frozen f=0,
f=1 and at the measured f) with both estimators, rebuild the prespecified agreement test
(run.py _agreement / _ratio_test / combine) independently for each estimator, and
refit the exploratory rounds 2-8 window. Writes v_agree.json.

Choices for the MLE version (mirroring what the refit report says it did):
  sigma_O  = half-width of the quasi-profile 68% interval (eps) or of the joint
             quasi-profile interval (Lambda, paired X jobs);
  sigma_sim = the same half-width for the prediction fit (200k shots);
  slope dM/df from the MLE-refit predictions at f=0, measured f, f=1 (parabola) or the
             line through f=0,1 when f is within 0.05 of an end or beyond;
  gap = |M - M_after| (0 throughout, as stored).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from v_fit import ROUNDS, eps_interval, lambda_profile, mle, wls

HERE = Path(__file__).resolve().parent
reps = json.loads((HERE / "v_counts.json").read_text())
res = json.loads((HERE / "v_results.json").read_text())
NS = 200_000


def counts(p, N):
    k = np.array([round(x * N) for x in p], float)
    assert np.allclose(k / N, p, atol=1e-9), p
    return k


def fit_both(p, N, n=ROUNDS):
    k = counts(p, N)
    w = wls(n, k / N, N)
    m = mle(k, N, n)
    qlo, qhi = eps_interval(k, N, m, m["phi"], n)
    return dict(wls=(w["eps"], w["se"]), mle=(m["eps"], (qhi - qlo) / 2), phi=m["phi"], k=k.tolist())


def lam_both(p3, p5, N, n=ROUNDS):
    k3, k5 = counts(p3, N), counts(p5, N)
    f3, f5 = mle(k3, N, n), mle(k5, N, n)
    lam, lo, hi = lambda_profile(k3, k5, N, N, f3["phi"], f5["phi"], n)
    w3, w5 = wls(n, k3 / N, N), wls(n, k5 / N, N)
    lw = w3["eps"] / w5["eps"]
    sw = lw * math.hypot(w3["se"] / w3["eps"], w5["se"] / w5["eps"])
    return dict(wls=(lw, sw), mle=(lam, (hi - lo) / 2))


def slope(f, v0, vf, v1):
    if 0.05 < f < 0.95:
        at = [0.0, f, 1.0]
        c = np.polyfit(at, [v0, vf, v1], 2)
    else:
        c = np.polyfit([0.0, 1.0], [v0, v1], 1)
    return float(np.polyval(np.polyder(c), f))


out = {"reps": []}
for r, fr in zip(reps, res):
    entry = {"day": r["day"], "rep": r["rep"], "tests": {"wls": {}, "mle": {}}, "pred": {}, "window": {}}
    jobs = {}
    for key, c in r["curves"].items():
        jobs.setdefault(c["folder"], []).append(key)
    # eps tests per curve
    for key, c in r["curves"].items():
        b, d = key[0], key[1]
        pm = fit_both(c["pred_measured_f"], NS)
        p0 = fit_both(c["pred_f0"], NS)
        p1 = fit_both(c["pred_f1"], NS)
        pa = fit_both(c["pred_after"], NS) if c["pred_after"] else None
        entry["pred"][key] = dict(measured=pm, f0=p0, f1=p1)
        # check stored prediction WLS fit reproduces
        assert abs(pm["wls"][0] - c["stored_pred_fit"][0]) < 1e-12, (key, pm["wls"], c["stored_pred_fit"])
        cv = fr["curves"][key]
        for est in ("wls", "mle"):
            if est == "wls":
                O, sO = cv["wls"]["eps"], cv["wls"]["se"]
            else:
                O, sO = cv["mle"]["eps"], cv["mle"]["sigma_q"]
            M, sM = pm[est]
            sl = slope(c["f"], p0[est][0], M, p1[est][0])
            from_f = sl * c["sigma_f"]
            gap = abs(M - pa[est][0]) if pa else 0.0
            bound = 2 * math.sqrt(sO**2 + sM**2 + from_f**2 + gap**2 / 2)
            entry["tests"][est][f"{b} eps{d}"] = dict(observed=O, predicted=M, sigma_O=sO, sim=sM, from_f=from_f,
                                                      gap=gap, bound=bound, slope=sl, sigma_f=c["sigma_f"],
                                                      folder=c["folder"])
        if est == "mle":
            pass
    # Lambda tests
    for b in "XZ":
        c3, c5 = r["curves"][f"{b}3"], r["curves"][f"{b}5"]
        paired = c3["folder"] == c5["folder"]
        lp = {}
        if paired and b == "X":
            # direct Lambda test from the job's own fits (run.py _quantities adds X lambda for paired jobs)
            lm = lam_both(c3["pred_measured_f"], c5["pred_measured_f"], NS)
            l0 = lam_both(c3["pred_f0"], c5["pred_f0"], NS)
            l1 = lam_both(c3["pred_f1"], c5["pred_f1"], NS)
            for est in ("wls", "mle"):
                L = fr["lam"][b]
                O, sO = (L["wls"], L["wls_se"]) if est == "wls" else (L["mle"], L["sigma_q"])
                M, sM = lm[est]
                sl = slope(c3["f"], l0[est][0], M, l1[est][0])
                from_f = sl * c3["sigma_f"]
                bound = 2 * math.sqrt(sO**2 + sM**2 + from_f**2)
                entry["tests"][est][f"{b} lambda"] = dict(observed=O, predicted=M, sigma_O=sO, sim=sM,
                                                          from_f=from_f, gap=0.0, bound=bound)
        else:
            for est in ("wls", "mle"):
                e3, e5 = entry["tests"][est][f"{b} eps3"], entry["tests"][est][f"{b} eps5"]
                O = e3["observed"] / e5["observed"]
                M = e3["predicted"] / e5["predicted"]

                def spread(key, against):
                    return math.sqrt(sum((row[key] / row[against]) ** 2 for row in (e3, e5)))

                sO = O * spread("sigma_O", "observed")
                if est == "mle":
                    # the joint profile half-width, as for paired jobs
                    sO = fr["lam"][b]["sigma_q"]
                sim = M * spread("sim", "predicted")
                ff = M * spread("from_f", "predicted")
                gap = M * spread("gap", "predicted")
                bound = 2 * math.sqrt(sO**2 + sim**2 + ff**2 + gap**2 / 2)
                entry["tests"][est][f"{b} lambda"] = dict(observed=O, predicted=M, sigma_O=sO, sim=sim,
                                                          from_f=ff, gap=gap, bound=bound)
    # window rounds 2-8 (exploratory)
    n2 = ROUNDS[1:]
    for b in "XZ":
        win = {}
        for d in "35":
            c = r["curves"][f"{b}{d}"]
            k = np.array(c["k"][1:], float)
            w = wls(n2, k / c["N"], c["N"])
            m = mle(k, c["N"], n2)
            win[d] = dict(wls=w["eps"], mle=m["eps"], stored=c["stored_window"][0])
            assert abs(w["eps"] - c["stored_window"][0]) < 1e-12
            pk = counts(c["pred_measured_f"][1:], NS)
            win[d]["pred_wls"] = wls(n2, pk / NS, NS)["eps"]
            win[d]["pred_mle"] = mle(pk, NS, n2)["eps"]
        entry["window"][b] = dict(
            wls=win["3"]["wls"] / win["5"]["wls"], mle=win["3"]["mle"] / win["5"]["mle"],
            pred_wls=win["3"]["pred_wls"] / win["5"]["pred_wls"], pred_mle=win["3"]["pred_mle"] / win["5"]["pred_mle"])
    out["reps"].append(entry)
    print(r["day"], r["rep"], flush=True)

# chi2 over reps

summary = {}
for est in ("wls", "mle"):
    for q in ("X lambda", "X eps3", "X eps5", "Z lambda", "Z eps3", "Z eps5"):
        z = [(e["tests"][est][q]["observed"] - e["tests"][est][q]["predicted"]) / (e["tests"][est][q]["bound"] / 2)
             for e in out["reps"]]
        summary[f"{est} {q}"] = float(sum(v * v for v in z))
    agree_all = [all(abs(e["tests"][est][q]["observed"] - e["tests"][est][q]["predicted"]) < e["tests"][est][q]["bound"]
                     for q in ("X lambda", "X eps3", "X eps5")) for e in out["reps"]]
    summary[f"{est} X all-three agree reps"] = [f"{e['day']} r{e['rep']}" for e, a in zip(out["reps"], agree_all) if a]
out["chi2"] = summary
print(json.dumps(summary, indent=1))

# pooled prediction with the observed-sigma weights, and unweighted means of predicted Lambda
for est in ("wls", "mle"):
    w = np.array([1 / e["tests"][est]["X lambda"]["sigma_O"] ** 2 for e in out["reps"]])
    pr = np.array([e["tests"][est]["X lambda"]["predicted"] for e in out["reps"]])
    prz = np.array([e["tests"][est]["Z lambda"]["predicted"] for e in out["reps"]])
    print(est, "pooled predicted Lambda_X (obs weights):", round(float((w * pr).sum() / w.sum()), 4),
          " unweighted mean predicted Lambda_X:", round(float(pr.mean()), 4), " Lambda_Z:", round(float(prz.mean()), 4))
# prediction refit shifts
de = max(abs(e["pred"][k]["measured"]["mle"][0] - e["pred"][k]["measured"]["wls"][0]) for e in out["reps"] for k in e["pred"])
dl = max(abs(e["tests"]["mle"][q]["predicted"] - e["tests"]["wls"][q]["predicted"]) for e in out["reps"] for q in ("X lambda", "Z lambda"))
print(f"max |MLE - WLS| on predictions: eps {de*100:.4f} pp, Lambda {dl:.4f}")
print("window Lambda_X:")
for est in ("wls", "mle", "pred_wls", "pred_mle"):
    for p in (1, 2):
        print(" ", est, "placement", p, [round(e["window"]["X"][est], 2) for e in out["reps"] if e["rep"] == p])
(HERE / "v_agree.json").write_text(json.dumps(out, indent=1))
