"""Refit all 40 hardware logical-error curves, and the stored model predictions, with the
prespecified weighted-log fit and with a binomial maximum-likelihood fit at every round count.

    python docs/diagnosis/revision/refits/extract_counts.py   # once
    python docs/diagnosis/revision/refits/refit.py [--boot 2000]

Reads counts.json (written by extract_counts.py from the run folders' analysis.json and
checked against a re-decode of shots.npz). Writes refits.json and per_rep.csv beside this
file. Exploratory, post hoc: nothing here changes the prespecified analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zlib
from math import sqrt
from pathlib import Path

import numpy as np
from scipy.stats import chi2
from scipy.stats import t as student_t

import fitlib as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
STOCKS = ("X", "Z")
PRED = "predicted_at_measured_dephasing"


def half(lo, hi):
    return (hi - lo) / 2


# ----------------------------------------------------------------------------- fits
def fit_curve(rounds, k, N):
    k = np.asarray(k, float)
    w = F.wls(rounds, k / N, N)
    b = F.binomial(rounds, k, N)
    return {
        "k": [int(x) for x in k],
        "N": N,
        "P": (k / N).tolist(),
        "points_at_or_above_half": int((k / N >= 0.5).sum()),
        "wls": {"eps": w.eps, "se": w.se, "se_shot_noise": w.se_raw, "chi2_red": w.chi2_red,
                "A": w.A, "points_used": w.used},
        "binomial": {"eps": b.eps, "A": b.A, "deviance": b.deviance, "p_gof": b.p_gof,
                     "phi": b.pearson_red, "lo68": b.lo, "hi68": b.hi,
                     "lo68_quasi": b.lo_q, "hi68_quasi": b.hi_q,
                     "se_quasi": half(b.lo_q, b.hi_q), "se": half(b.lo, b.hi),
                     "upper_at_boundary": b.at_boundary},
    }


def fit_pred(rounds, P, N):
    """A stored prediction (P per round from N simulated shots), fitted both ways."""
    k = np.round(np.asarray(P) * N)
    return fit_curve(rounds, k, N)


def bootstrap(rounds, fit, B, seed):
    """Parametric bootstrap from the binomial MLE curve: refit both estimators."""
    n = np.asarray(rounds, float)
    N = fit["N"]
    eps, A = fit["binomial"]["eps"], fit["binomial"]["A"]
    P = (1 - A * (1 - 2 * eps) ** n) / 2
    rng = np.random.default_rng(seed)
    k = rng.binomial(N, P, size=(B, len(n))).astype(float)
    grid = np.concatenate([np.linspace(1e-4, 0.45, 240), np.linspace(0.45, F.EPS_MAX, 40)[1:]])
    out_m, out_w = [], []
    for s in range(0, B, 250):
        out_m.append(F.mle_batch(k[s:s + 250], N, n, grid=grid)[0])
        out_w.append(F.wls_batch(n, k[s:s + 250] / N, N)[0])
    return np.concatenate(out_m), np.concatenate(out_w)


def pct(x, q=(15.865, 84.135)):
    lo, hi = np.percentile(x, q)
    return float(lo), float(hi)


# ------------------------------------------------- prespecified agreement, re-implemented
def quantities(fits, basis):
    found = {f"eps{d}": tuple(v) for d, v in fits[basis]["per_round"].items()}
    if basis == "X" and "lambda" in fits[basis]:
        found["lambda"] = (fits[basis]["lambda"], fits[basis]["lambda_uncertainty"])
    return found


def fits_struct(curves_by_basis, est, lam_override=None):
    """The repo's fits layout from fitted curves: per_round {d: [eps, se]} and Λ if both.

    est 'wls': se is the widened se; Λ σ by propagation, as prespecified.
    est 'binomial': se is the quasi-likelihood profile half-width; Λ σ by propagation
    unless lam_override gives (Λ, σ) per basis.
    """
    out = {}
    for b, curves in curves_by_basis.items():
        pr = {}
        for d, c in curves.items():
            if est == "wls":
                pr[d] = [c["wls"]["eps"], c["wls"]["se"]]
            else:
                pr[d] = [c["binomial"]["eps"], c["binomial"]["se_quasi"]]
        out[b] = {"per_round": pr}
        if {"3", "5"} <= set(pr):
            (e3, u3), (e5, u5) = pr["3"], pr["5"]
            lam = e3 / e5
            sig = lam * sqrt((u3 / e3) ** 2 + (u5 / e5) ** 2)
            if lam_override and b in lam_override:
                lam, sig = lam_override[b]
            out[b].update({"lambda": lam, "lambda_uncertainty": sig})
    return out


def slopes(frozen0, frozen1, expected, f):
    known = [(0.0, frozen0), (1.0, frozen1)]
    if 0.05 < f < 0.95:
        known.insert(1, (f, expected))
    at = np.array([x for x, _ in known])
    found = {}
    for b in STOCKS:
        values = [quantities(fits, b) for _, fits in known]
        for name in values[0]:
            curve = np.polyfit(at, [v[name][0] for v in values], len(known) - 1)
            found[f"{b} {name}"] = float(np.polyval(np.polyder(curve), f))
    return found


def agreement(observed, predicted, after, slope, sigma_f):
    found = {}
    for b in STOCKS:
        seen, want = quantities(observed, b), quantities(predicted, b)
        later = quantities(after, b) if after else {}
        for name, (value, sigma) in seen.items():
            guess, noise = want[name]
            gap = abs(guess - later[name][0]) if later else 0.0
            from_f = slope.get(f"{b} {name}", 0.0) * sigma_f
            bound = 2 * sqrt(sigma**2 + noise**2 + from_f**2 + gap**2 / 2)
            found[f"{b} {name}"] = {"observed": value, "predicted": guess,
                                    "observed_uncertainty": sigma, "simulator_noise": noise,
                                    "from_f": from_f, "calibration_gap": gap, "bound": bound}
    return found


def ratio_test(e3, e5):
    observed = e3["observed"] / e5["observed"]
    predicted = e3["predicted"] / e5["predicted"]

    def spread(key, against):
        return sqrt(sum((row[key] / row[against]) ** 2 for row in (e3, e5)))

    sigma = observed * spread("observed_uncertainty", "observed")
    noise = predicted * spread("simulator_noise", "predicted")
    from_f = predicted * spread("from_f", "predicted")
    gap = predicted * spread("calibration_gap", "predicted")
    bound = 2 * sqrt(sigma**2 + noise**2 + from_f**2 + gap**2 / 2)
    return {"observed": observed, "predicted": predicted, "observed_uncertainty": sigma,
            "simulator_noise": noise, "from_f": from_f, "calibration_gap": gap, "bound": bound}


# --------------------------------------------------------------------------- pooling
def pooled(values, sigmas):
    """Inverse-variance mean, se widened by sqrt(chi2/dof) when > 1 (the prespecified rule)."""
    v, w = np.asarray(values, float), 1 / np.asarray(sigmas, float) ** 2
    m = float((w * v).sum() / w.sum())
    k = len(v)
    spread = float((w * (v - m) ** 2).sum() / (k - 1)) if k > 1 else 1.0
    return m, float(sqrt(max(spread, 1.0) / w.sum())), spread


def decide(lam, sigma, critical):
    if lam - critical * sigma > 1:
        return "d = 5 better"
    if lam + critical * sigma < 1:
        return "d = 5 worse"
    return "undecided"


# ------------------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default=str(HERE), help="directory for refits.json and per_rep.csv")
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    out_dir = Path(args.out)
    counts = json.loads((HERE / "counts.json").read_text())
    rounds = tuple(counts["rounds"])
    jobs = counts["jobs"]

    # ---- hardware curves and per-job predictions
    obs, pred, boot = {}, {}, {}
    reps: dict = {}
    for name, j in jobs.items():
        key = f"{j['day']} r{j['rep']}"
        reps.setdefault(key, {"day": j["day"], "rep": j["rep"], "pairing": j["pairing"], "jobs": {}})
        pred[name] = {}
        for p in j["patches"]:
            reps[key]["jobs"][p] = name
        for b in STOCKS:
            for p in j["patches"]:
                c = fit_curve(rounds, j["observed"][b][p], j["shots"])
                reg = j["registered_fit"][b]["per_round"][p]
                c["registered"] = {"eps": reg[0], "se": reg[1]}
                assert abs(c["wls"]["eps"] - reg[0]) < 1e-12 and abs(c["wls"]["se"] - reg[1]) < 1e-12
                obs[(key, b, p)] = c
                seed = zlib.crc32(f"{key}{b}{p}".encode())
                bm, bw = bootstrap(rounds, c, args.boot, seed)
                boot[(key, b, p)] = (bm, bw)
                eh = c["binomial"]["eps"]
                c["bootstrap"] = {
                    "B": args.boot, "binomial_lo68": pct(bm)[0], "binomial_hi68": pct(bm)[1],
                    "binomial_sd": float(bm.std(ddof=1)), "wls_sd": float(bw.std(ddof=1)),
                    "binomial_bias": float(bm.mean() - eh), "wls_bias": float(bw.mean() - eh),
                    "wls_median_bias": float(np.median(bw) - eh),
                }
                print(f"{key} {b} d{p}: wls {100*c['wls']['eps']:.2f}±{100*c['wls']['se']:.2f}  "
                      f"mle {100*eh:.2f} [{100*c['binomial']['lo68']:.2f},{100*c['binomial']['hi68']:.2f}]"
                      f" quasi [{100*c['binomial']['lo68_quasi']:.2f},{100*c['binomial']['hi68_quasi']:.2f}]"
                      f" boot [{100*pct(bm)[0]:.2f},{100*pct(bm)[1]:.2f}]  dev {c['binomial']['deviance']:.1f}"
                      f"  >=0.5 pts {c['points_at_or_above_half']}  wls bias {100*c['bootstrap']['wls_bias']:+.2f}",
                      flush=True)
        for src, s in j["predicted"].items():
            pred[name][src] = {b: {p: fit_pred(rounds, s["P"][b][p], s["shots"]) for p in j["patches"]}
                               for b in STOCKS}

    # ---- Λ per rep and basis
    per_rep = {}
    for key, r in sorted(reps.items()):
        row = {"day": r["day"], "rep": r["rep"], "placement": r["rep"], "pairing": r["pairing"],
               "jobs": r["jobs"]}
        for b in STOCKS:
            c3, c5 = obs[(key, b, "3")], obs[(key, b, "5")]
            w3, w5 = c3["wls"], c5["wls"]
            lw = w3["eps"] / w5["eps"]
            sw = lw * sqrt((w3["se"] / w3["eps"]) ** 2 + (w5["se"] / w5["eps"]) ** 2)
            lam, lo, hi = F.lambda_profile(rounds, c3["k"], c3["N"], c5["k"], c5["N"])
            phi3 = max(c3["binomial"]["phi"], 1.0)
            phi5 = max(c5["binomial"]["phi"], 1.0)
            _, loq, hiq = F.lambda_profile(rounds, c3["k"], c3["N"], c5["k"], c5["N"], phi3, phi5)
            b3, b5 = c3["binomial"], c5["binomial"]
            prop = lam * sqrt((b3["se_quasi"] / b3["eps"]) ** 2 + (b5["se_quasi"] / b5["eps"]) ** 2)
            (m3, ww3), (m5, ww5) = boot[(key, b, "3")], boot[(key, b, "5")]
            lm, lwb = m3 / m5, ww3 / ww5
            # predictions (each patch from its own job)
            P3 = pred[r["jobs"]["3"]][PRED][b]["3"]
            P5 = pred[r["jobs"]["5"]][PRED][b]["5"]
            row[b] = {
                "eps3": {"wls": [w3["eps"], w3["se"]], "binomial": b3["eps"],
                         "binomial_lo68": b3["lo68"], "binomial_hi68": b3["hi68"],
                         "binomial_lo68_quasi": b3["lo68_quasi"], "binomial_hi68_quasi": b3["hi68_quasi"],
                         "bootstrap68": [c3["bootstrap"]["binomial_lo68"], c3["bootstrap"]["binomial_hi68"]]},
                "eps5": {"wls": [w5["eps"], w5["se"]], "binomial": b5["eps"],
                         "binomial_lo68": b5["lo68"], "binomial_hi68": b5["hi68"],
                         "binomial_lo68_quasi": b5["lo68_quasi"], "binomial_hi68_quasi": b5["hi68_quasi"],
                         "bootstrap68": [c5["bootstrap"]["binomial_lo68"], c5["bootstrap"]["binomial_hi68"]]},
                "lambda": {"wls": [lw, sw], "binomial": lam, "binomial_lo68": lo, "binomial_hi68": hi,
                           "binomial_lo68_quasi": loq, "binomial_hi68_quasi": hiq,
                           "binomial_se_quasi": half(loq, hiq), "binomial_se_propagated": prop,
                           "bootstrap68": list(pct(lm)), "bootstrap_sd": float(lm.std(ddof=1)),
                           "bootstrap_binomial_bias": float(lm.mean() - lam),
                           "bootstrap_wls_bias": float(lwb.mean() - lam),
                           "bootstrap_wls_sd": float(lwb.std(ddof=1))},
                "predicted": {
                    "eps3": {"wls": P3["wls"]["eps"], "binomial": P3["binomial"]["eps"]},
                    "eps5": {"wls": P5["wls"]["eps"], "binomial": P5["binomial"]["eps"]},
                    "lambda": {"wls": P3["wls"]["eps"] / P5["wls"]["eps"],
                               "binomial": P3["binomial"]["eps"] / P5["binomial"]["eps"]},
                },
            }
            # exploratory window, rounds 2-8, as in the prespecified from_round_2
            keep = [i for i, n in enumerate(rounds) if n >= 2]
            rw = tuple(rounds[i] for i in keep)
            k3w, k5w = [c3["k"][i] for i in keep], [c5["k"][i] for i in keep]
            e3w, e5w = F.wls(rw, np.array(k3w) / c3["N"], c3["N"]), F.wls(rw, np.array(k5w) / c5["N"], c5["N"])
            lamw, low, hiw = F.lambda_profile(rw, k3w, c3["N"], k5w, c5["N"])
            row[b]["from_round_2"] = {"wls": e3w.eps / e5w.eps, "binomial": lamw,
                                      "binomial_lo68": low, "binomial_hi68": hiw}
        per_rep[key] = row

    keys = sorted(per_rep)

    # ---- summaries
    def collect(b, q, est):
        out = []
        for k in keys:
            x = per_rep[k][b][q]
            if est == "wls":
                out.append(tuple(x["wls"]))
            elif q == "lambda":
                out.append((x["binomial"], x["binomial_se_quasi"]))
            else:
                out.append((x["binomial"], half(x["binomial_lo68_quasi"], x["binomial_hi68_quasi"])))
        return np.array(out)

    summary: dict = {"unweighted_means": {}, "placement_pooled": {}, "pooled_lambda": {},
                     "predicted_means": {}}
    for b in STOCKS:
        for q in ("eps3", "eps5", "lambda"):
            for est in ("wls", "binomial"):
                v = collect(b, q, est)[:, 0]
                summary["unweighted_means"][f"{b} {q} {est}"] = {
                    "mean": float(v.mean()), "se": float(v.std(ddof=1) / sqrt(len(v))),
                    "min": float(v.min()), "max": float(v.max())}
                pv = np.array([per_rep[k][b]["predicted"][q][est] for k in keys])
                summary["predicted_means"][f"{b} {q} {est}"] = {
                    "mean": float(pv.mean()),
                    **{f"placement {pl}": float(pv[[per_rep[k]['placement'] == pl for k in keys]].mean())
                       for pl in (1, 2)},
                    **{f"placement {pl} range": [float(pv[[per_rep[k]['placement'] == pl for k in keys]].min()),
                                                 float(pv[[per_rep[k]['placement'] == pl for k in keys]].max())]
                       for pl in (1, 2)}}
                for pl in (1, 2):
                    ix = [per_rep[k]["placement"] == pl for k in keys]
                    vals = collect(b, q, est)[ix]
                    m, se, spread = pooled(vals[:, 0], vals[:, 1])
                    summary["placement_pooled"][f"{b} {q} {est} placement {pl}"] = {
                        "pooled": m, "se_widened": se, "chi2_per_dof": spread, "days": int(sum(ix))}

    for b in STOCKS:
        for est in ("wls", "binomial"):
            vals = collect(b, "lambda", est)
            lam, sig, spread = pooled(vals[:, 0], vals[:, 1])
            crit = float(student_t.ppf(0.95, len(vals) - 1))
            predicted = np.array([per_rep[k][b]["predicted"]["lambda"][est] for k in keys])
            w = 1 / vals[:, 1] ** 2
            pl = [pooled(*collect(b, "lambda", est)[[per_rep[k]["placement"] == p for k in keys]].T)[0]
                  for p in (1, 2)]
            pm, pse = float(np.mean(pl)), float(np.std(pl, ddof=1) / sqrt(2))
            tval = (1 - pm) / pse
            ln_m, ln_se, _ = pooled(np.log(vals[:, 0]), vals[:, 1] / vals[:, 0])
            summary["pooled_lambda"][f"{b} {est}"] = {
                "lambda": lam, "se_widened": sig, "sqrt_chi2_per_dof": sqrt(spread),
                "sigmas_below_one": (1 - lam) / sig, "t9_critical": crit,
                "verdict": decide(lam, sig, crit),
                "predicted_weighted": float((w * predicted).sum() / w.sum()),
                "placement_as_unit": {"placements": pl, "mean": pm, "se": pse,
                                      "one_sided_p_lambda_below_1": float(student_t.sf(tval, 1))},
                "ln_pooled": {"lambda": float(np.exp(ln_m)), "low": float(np.exp(ln_m - ln_se)),
                              "high": float(np.exp(ln_m + ln_se))},
                "unweighted": summary["unweighted_means"][f"{b} lambda {est}"],
            }

    # ---- the prespecified agreement test, then the same with the binomial fit throughout
    agreement_out = {}
    for est in ("wls", "binomial"):
        tests_by_rep = {}
        for key in keys:
            r = reps[key]
            tests = {}
            for name in dict.fromkeys(r["jobs"].values()):
                j = jobs[name]
                ob = {b: {p: obs[(key, b, p)] for p in j["patches"]} for b in STOCKS}
                lam_ovr = None
                if est == "binomial" and len(j["patches"]) == 2:
                    lam_ovr = {b: (per_rep[key][b]["lambda"]["binomial"],
                                   per_rep[key][b]["lambda"]["binomial_se_quasi"]) for b in STOCKS}
                observed = fits_struct(ob, est, lam_ovr)
                fs = {src: fits_struct(pred[name][src], est) for src in pred[name]}
                sl = slopes(fs["frozen_f0.0"], fs["frozen_f1.0"], fs[PRED], j["f"])
                tests.update(agreement(observed, fs[PRED], fs["predicted_from_calibration_after"],
                                       sl, j["f_uncertainty"]))
            for b in STOCKS:
                if f"{b} lambda" not in tests:
                    tests[f"{b} lambda"] = ratio_test(tests[f"{b} eps3"], tests[f"{b} eps5"])
                    if est == "binomial":  # observed Λ and σ from the joint profile, as above
                        t = tests[f"{b} lambda"]
                        t["observed"] = per_rep[key][b]["lambda"]["binomial"]
                        t["observed_uncertainty"] = per_rep[key][b]["lambda"]["binomial_se_quasi"]
                        t["bound"] = 2 * sqrt(t["observed_uncertainty"] ** 2 + t["simulator_noise"] ** 2
                                              + t["from_f"] ** 2 + t["calibration_gap"] ** 2 / 2)
            for t in tests.values():
                t["agrees"] = abs(t["observed"] - t["predicted"]) < t["bound"]
            tests_by_rep[key] = tests
        model = {}
        for q in ("X lambda", "X eps3", "X eps5", "Z lambda", "Z eps3", "Z eps5"):
            z = [(t[q]["observed"] - t[q]["predicted"]) / (t[q]["bound"] / 2) for t in tests_by_rep.values()]
            stat = float(sum(v * v for v in z))
            bar = float(chi2.ppf(0.95, len(z)))
            model[q] = {"chi2": stat, "bar": bar, "p_value": float(chi2.sf(stat, len(z))),
                        "agrees": stat < bar,
                        "reps_agreeing": int(sum(t[q]["agrees"] for t in tests_by_rep.values()))}
        agreement_out[est] = {"model": model, "tests": tests_by_rep}

    # check the WLS replication against runs/combined.json
    combined = json.loads((REPO / "runs/combined.json").read_text())
    check = {q: {"registered": combined["model"][q]["chi2"], "replicated": agreement_out["wls"]["model"][q]["chi2"]}
             for q in combined["model"]}
    check["pooled_x"] = {"registered": [combined["pooled_x"]["lambda"], combined["pooled_x"]["lambda_uncertainty"]],
                         "replicated": [summary["pooled_lambda"]["X wls"]["lambda"],
                                        summary["pooled_lambda"]["X wls"]["se_widened"]]}

    # ---- all prediction curves (every stored source), both estimators
    pred_out = {name: {src: {b: {p: {"wls": c["wls"], "binomial": c["binomial"], "P": c["P"], "N": c["N"]}
                                 for p, c in v.items()} for b, v in s.items()}
                       for src, s in srcs.items()} for name, srcs in pred.items()}

    curves = {f"{k} {b} d{p}": c for (k, b, p), c in obs.items()}
    gof = {"deviance_sum": float(sum(c["binomial"]["deviance"] for c in obs.values())),
           "dof_sum": 4 * len(obs),
           "curves_p_below_0.05": [k for k, c in curves.items() if c["binomial"]["p_gof"] < 0.05],
           "curves_with_points_at_or_above_half": {k: c["points_at_or_above_half"] for k, c in curves.items()
                                                   if c["points_at_or_above_half"]},
           "A_above_1": {k: c["binomial"]["A"] for k, c in curves.items() if c["binomial"]["A"] > 1},
           "upper_at_boundary": [k for k, c in curves.items() if c["binomial"]["upper_at_boundary"]]}
    gof["p_total"] = float(chi2.sf(gof["deviance_sum"], gof["dof_sum"]))

    result = {
        "method": {
            "model": "P(n) = (1 - A (1 - 2 eps)^n) / 2, n = " + str(rounds),
            "wls": "registered fit (heavyhex.simulation.noisy.fit_per_round), re-implemented; "
                   "matches the stored values to 1e-12",
            "binomial": "k_n ~ Binomial(N, P(n)) at all six n; A profiled exactly (concave) on [0, 1/(1-2eps)); "
                        "68.27% profile-likelihood interval where 2 (lmax - l(eps)) <= 1; 'quasi' interval uses "
                        "<= phi with phi = max(1, Pearson X^2/4), the analogue of the registered sqrt(chi2_red) widening; "
                        "se = half-width of an interval",
            "lambda": "binomial Λ = eps3/eps5 from the joint likelihood of the two independent curves, profiled over "
                      "eps5 (quasi: each curve's log-likelihood divided by its phi); bootstrap: ratio of independent "
                      "replicates",
            "bootstrap": f"parametric, {args.boot} replicates per curve drawn from the binomial MLE curve; "
                         "68% percentile interval; bias = replicate mean - MLE value used to generate",
            "pooling": "inverse-variance mean, se widened by sqrt(chi2/dof) if > 1, Student t(9) bar (registered rule); "
                       "binomial uses quasi half-widths as sigmas",
            "predictions": "stored per-round P(n) from analysis.json (200k simulated shots; each-qubit f 50k) and "
                           "prediction.json (f=0, f=1; 200k), counts = round(P N), fitted with both estimators",
        },
        "replication_check": check,
        "per_rep": per_rep,
        "summary": summary,
        "agreement": {est: v["model"] for est, v in agreement_out.items()},
        "agreement_tests": {est: v["tests"] for est, v in agreement_out.items()},
        "goodness_of_fit": gof,
        "curves": curves,
        "predictions": pred_out,
    }
    (out_dir / "refits.json").write_text(json.dumps(result, indent=1) + "\n", encoding="utf-8")

    with open(out_dir / "per_rep.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["day", "rep", "basis", "quantity", "wls", "wls_se", "binomial", "prof_lo68", "prof_hi68",
                     "quasi_lo68", "quasi_hi68", "boot_lo68", "boot_hi68", "pred_wls", "pred_binomial"])
        for k in keys:
            r = per_rep[k]
            for b in STOCKS:
                for q in ("eps3", "eps5", "lambda"):
                    x = r[b][q]
                    wr.writerow([r["day"], r["rep"], b, q, x["wls"][0], x["wls"][1], x["binomial"],
                                 x["binomial_lo68"], x["binomial_hi68"], x["binomial_lo68_quasi"],
                                 x["binomial_hi68_quasi"], *x["bootstrap68"],
                                 r[b]["predicted"][q]["wls"], r[b]["predicted"][q]["binomial"]])

    import summarize

    summarize.main(out_dir / "refits.json")


if __name__ == "__main__":
    main()
