"""Independent verification of the heating analysis (../heating/), with its own statistics.

Reads v_per_shot.npz / v_meta.json (from v_extract.py). Writes v_heating.json.

Time: shot index j in 0..624 of each piece (the README's interleaved order, tested in
v_order.py). t_j = (j + 0.5) / 625 in [0, 1]; a slope is the fractional change from
start (t=0) to end (t=1) of the job's shots.

For each metric, per series s = (job, setting, patch): value per shot v (4 memory pieces or
2 idle pieces, stacked). Bins b = floor(10 j / 625). y_sb = mean of v in bin / series mean.

Per-job slope estimates (three error models, to test the stated SE):
  (a) WLS common slope, per-series intercepts, bin errors from within-bin per-shot
      variance; SE inflated by sqrt(chi2/dof) if > 1.            [same idea as theirs]
  (b) 'job curve': inverse-variance mean of y_sb over the job's series per bin ->
      10 points; OLS line; SE from residual scatter (8 dof) -- this captures any
      common-mode (all-series) fluctuation that (a) treats as independent.
  (c) permutation: shuffle shot order within each piece 400x; SE = sd of null slopes
      computed with the (a) point estimator (exchangeable-shot null, no assumptions
      about per-shot variance).
Pooled over 15 jobs: DerSimonian-Laird random effects with (a) and with max(a, b) SEs;
plain mean of 15 slopes with Student t (14 dof); sign count; per-job z.
"""

from __future__ import annotations

import json
from collections import defaultdict
from math import sqrt
from pathlib import Path

import numpy as np
from scipy import stats

from heavyhex.experiment.circuits import ROUNDS, SETTINGS
from heavyhex.simulation.noisy import fit_per_round

HERE = Path(__file__).resolve().parent
D = np.load(HERE / "v_per_shot.npz")
META = json.loads((HERE / "v_meta.json").read_text())
JOBS = list(META)
NB = 10
N = 625
J = np.arange(N)
BIN = (J * NB) // N
TB = np.array([((J[BIN == b] + 0.5) / N).mean() for b in range(NB)])
RNG = np.random.default_rng(12345)
BONF = float(stats.norm.isf(0.05 / 15 / 2))


def pieces_of(job, i):
    return sorted(int(k) for k, p in META[job]["pieces"].items() if p["i"] == i)


def mem_series(what, basis=None, patch=None):
    out = []
    for job in JOBS:
        for i, s in enumerate(SETTINGS):
            if s.kind != "memory" or (basis and s.basis != basis):
                continue
            if what == "flag" and s.basis != "Z":
                continue
            for p in META[job]["patches"]:
                if patch and p != patch:
                    continue
                V = np.stack([D[f"mem/{job}/{k}/{p}/{what}"] for k in pieces_of(job, i)]).astype(float)
                out.append((job, (i, p), V))
    return out


def idle_series(what, readout=None):
    out = []
    for job in JOBS:
        for i, s in enumerate(SETTINGS):
            if s.kind != "idle" or (readout and s.readout != readout):
                continue
            for p in META[job]["patches"]:
                if what == "anc":
                    V = np.stack([D[f"idle/{job}/{k}/{p}/anc"] for k in pieces_of(job, i)]).astype(float)
                else:
                    V = np.stack([D[f"idle/{job}/{k}/{p}/data"].sum(1) for k in pieces_of(job, i)]).astype(float)
                out.append((job, (i, p), V))
    return out


def bins_of(V):
    m = V.mean()
    if m <= 0:
        return None
    mu = np.array([V[:, BIN == b].mean() for b in range(NB)])
    n = np.array([V[:, BIN == b].size for b in range(NB)])
    # within-bin variance per bin (not pooled), then pooled (less noisy); use pooled
    resid = V - mu[BIN][None, :]
    var = (resid ** 2).sum() / (V.size - NB)
    if var <= 0:
        return None
    return mu / m, np.sqrt(var / n) / m


def wls_common(rows):
    """rows: list of (y, se) over NB bins at TB. Common slope, per-row intercept."""
    num = den = chi2 = 0.0
    dof = -1
    fits = []
    for y, se in rows:
        w = 1 / se ** 2
        tb = (w * TB).sum() / w.sum()
        yb = (w * y).sum() / w.sum()
        num += (w * (TB - tb) * (y - yb)).sum()
        den += (w * (TB - tb) ** 2).sum()
        fits.append((y, w, tb, yb))
        dof += NB - 1
    b = num / den
    for y, w, tb, yb in fits:
        chi2 += (w * (y - yb - b * (TB - tb)) ** 2).sum()
    red = chi2 / dof
    se = sqrt(1 / den)
    return b, se, red, se * sqrt(max(red, 1.0))


def job_curve(rows):
    Y = np.array([r[0] for r in rows])
    W = 1 / np.array([r[1] for r in rows]) ** 2
    c = (W * Y).sum(0) / W.sum(0)
    X = np.vstack([np.ones(NB), TB]).T
    beta, res, *_ = np.linalg.lstsq(X, c, rcond=None)
    resid = c - X @ beta
    s2 = (resid ** 2).sum() / (NB - 2)
    cov = s2 * np.linalg.inv(X.T @ X)
    return beta[1], sqrt(cov[1, 1]), c


def perm_se(Vs, nperm=300):
    """SD of the (a)-estimator slope with shot order shuffled independently in each piece."""
    sl = []
    for _ in range(nperm):
        rows = []
        for V in Vs:
            P = np.stack([RNG.permutation(v) for v in V])
            r = bins_of(P)
            if r is not None:
                rows.append(r)
        sl.append(wls_common(rows)[0])
    return float(np.std(sl, ddof=1))


def dl(s, e):
    s, e = np.asarray(s), np.asarray(e)
    w = 1 / e ** 2
    fe = (w * s).sum() / w.sum()
    fe_se = sqrt(1 / w.sum())
    q = (w * (s - fe) ** 2).sum()
    k = len(s)
    tau2 = max(0.0, (q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum()))
    wr = 1 / (e ** 2 + tau2)
    re = (wr * s).sum() / wr.sum()
    re_se = sqrt(1 / wr.sum())
    return {"fe": fe, "fe_se": fe_se, "re": re, "re_se": re_se, "re_ci": [re - 1.96 * re_se, re + 1.96 * re_se],
            "tau": sqrt(tau2), "Q": q, "Q_p": float(stats.chi2.sf(q, k - 1))}


def analyse(series, do_perm=False, name=""):
    by_job = defaultdict(list)
    Vs_by_job = defaultdict(list)
    for job, lab, V in series:
        r = bins_of(V)
        if r is None:
            continue
        by_job[job].append(r)
        Vs_by_job[job].append(V)
    per = {}
    for job in JOBS:
        if job not in by_job:
            continue
        rows = by_job[job]
        b, se0, red, se = wls_common(rows)
        cb, cse, curve = job_curve(rows)
        d = {"slope": b, "se_shot": se0, "chi2_red": red, "se": se, "z": b / se,
             "curve_slope": cb, "curve_se": cse, "series": len(rows)}
        if do_perm:
            d["perm_se"] = perm_se(Vs_by_job[job])
        per[job] = d
    s = np.array([per[j]["slope"] for j in per])
    e = np.array([per[j]["se"] for j in per])
    erob = np.array([max(per[j]["se"], per[j]["curve_se"]) for j in per])
    allrows = [r for j in by_job for r in by_job[j]]
    pooled = wls_common(allrows)
    k = len(s)
    tcrit = stats.t.ppf(0.975, k - 1)
    pm, pse = s.mean(), s.std(ddof=1) / sqrt(k)
    z = s / e
    out = {
        "pooled_wls": {"slope": pooled[0], "se": pooled[3], "chi2_red": pooled[2]},
        "dl_a": dl(s, e),
        "dl_robust": dl(s, erob),
        "plain_t": {"mean": pm, "se": pse, "ci": [pm - tcrit * pse, pm + tcrit * pse]},
        "positive": int((s > 0).sum()), "jobs": k,
        "z_min": float(z.min()), "z_max": float(z.max()),
        "bonf_rise": [j for j in per if per[j]["z"] > BONF],
        "bonf_fall": [j for j in per if per[j]["z"] < -BONF],
        "per_job": per,
        "mean_ratio_curve_se_over_a_se": float(np.mean([per[j]["curve_se"] / per[j]["se"] for j in per])),
        "median_ratio_curve_se_over_a_se": float(np.median([per[j]["curve_se"] / per[j]["se"] for j in per])),
    }
    if do_perm:
        out["mean_ratio_perm_se_over_a_se"] = float(np.mean([per[j]["perm_se"] / per[j]["se"] for j in per]))
        out["mean_ratio_perm_se_over_shot_se"] = float(np.mean([per[j]["perm_se"] / per[j]["se_shot"] for j in per]))
    # curve-based meta (job curve slopes)
    cs = np.array([per[j]["curve_slope"] for j in per])
    ce = np.array([per[j]["curve_se"] for j in per])
    out["dl_curve"] = dl(cs, ce)
    line = (f"{name:28s} RE(a) {100*out['dl_a']['re']:+.2f} [{100*out['dl_a']['re_ci'][0]:+.2f},{100*out['dl_a']['re_ci'][1]:+.2f}]"
            f" FE {100*out['dl_a']['fe']:+.2f}±{100*out['dl_a']['fe_se']:.2f}"
            f" | RE(robust) {100*out['dl_robust']['re']:+.2f} [{100*out['dl_robust']['re_ci'][0]:+.2f},{100*out['dl_robust']['re_ci'][1]:+.2f}]"
            f" | t {100*pm:+.2f} [{100*out['plain_t']['ci'][0]:+.2f},{100*out['plain_t']['ci'][1]:+.2f}]"
            f" | pos {out['positive']}/{k} z[{out['z_min']:+.2f},{out['z_max']:+.2f}] tau {100*out['dl_a']['tau']:.2f}"
            f" | curveSE/aSE med {out['median_ratio_curve_se_over_a_se']:.2f}")
    if do_perm:
        line += f" permSE/aSE {out['mean_ratio_perm_se_over_a_se']:.2f} permSE/shotSE {out['mean_ratio_perm_se_over_shot_se']:.2f}"
    print(line, flush=True)
    return out


def eps_analysis():
    """eps per round per bin via the repo's fit_per_round; per (job, basis, patch)."""
    by_job = defaultdict(list)
    each = []
    for job in JOBS:
        for basis in "XZ":
            idx = [i for i, s in enumerate(SETTINGS) if s.kind == "memory" and s.basis == basis]
            for p in META[job]["patches"]:
                F = [np.stack([D[f"mem/{job}/{k}/{p}/fail"] for k in pieces_of(job, i)]).astype(float) for i in idx]
                eps, unc = [], []
                for b in range(NB):
                    nshot = F[0][:, BIN == b].size
                    fit = fit_per_round(ROUNDS, [f[:, BIN == b].mean() for f in F], nshot)
                    eps.append(fit.per_round)
                    unc.append(fit.uncertainty)
                eps, unc = np.array(eps), np.array(unc)
                ok = np.isfinite(unc) & (unc > 0)
                # this verification's choice: per-bin own errors (not averaged), normalized by the full-job fit
                full = fit_per_round(ROUNDS, [f.mean() for f in F], F[0].size).per_round
                y = eps[ok] / full
                se = unc[ok] / full
                t = TB[ok]
                w = 1 / se ** 2
                tb = (w * t).sum() / w.sum()
                yb = (w * y).sum() / w.sum()
                Sxx = (w * (t - tb) ** 2).sum()
                bslope = (w * (t - tb) * (y - yb)).sum() / Sxx
                # also unweighted OLS with residual scatter (robust to bad per-bin errors)
                X = np.vstack([np.ones(ok.sum()), t]).T
                beta = np.linalg.lstsq(X, eps[ok] / eps[ok].mean(), rcond=None)[0]
                r = eps[ok] / eps[ok].mean() - X @ beta
                s2 = (r ** 2).sum() / (ok.sum() - 2)
                ols_se = sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
                by_job[job].append((beta[1], ols_se))
                each.append({"job": job, "basis": basis, "patch": p, "ols_slope": beta[1], "ols_se": ols_se,
                             "wls_slope": bslope, "wls_se": sqrt(1 / Sxx), "bins_ok": int(ok.sum()),
                             "eps_full": full, "eps_bin_mean": float(eps[ok].mean())})
    per = {}
    for job, rows in by_job.items():
        s = np.array([r[0] for r in rows])
        e = np.array([r[1] for r in rows])
        w = 1 / e ** 2
        per[job] = {"slope": float((w * s).sum() / w.sum()), "se": float(sqrt(1 / w.sum()))}
    s = np.array([per[j]["slope"] for j in per])
    e = np.array([per[j]["se"] for j in per])
    k = len(s)
    pm, pse = s.mean(), s.std(ddof=1) / sqrt(k)
    tcrit = stats.t.ppf(0.975, k - 1)
    ser = np.array([r["ols_slope"] for r in each])
    out = {"dl": dl(s, e), "plain_t": {"mean": pm, "ci": [pm - tcrit * pse, pm + tcrit * pse]},
           "positive_jobs": int((s > 0).sum()), "series_positive": int((ser > 0).sum()), "series": len(ser),
           "z_min": float((s / e).min()), "z_max": float((s / e).max()), "per_job": per, "series_rows": each,
           "bin_fit_low_bias_mean": float(np.mean([r["eps_bin_mean"] / r["eps_full"] for r in each]))}
    print(f"eps/bin (OLS per series, IV per job) RE {100*out['dl']['re']:+.2f} [{100*out['dl']['re_ci'][0]:+.2f},{100*out['dl']['re_ci'][1]:+.2f}]"
          f" t {100*pm:+.2f} [{100*out['plain_t']['ci'][0]:+.2f},{100*out['plain_t']['ci'][1]:+.2f}]"
          f" pos jobs {out['positive_jobs']}/{k}, pos series {out['series_positive']}/{len(ser)};"
          f" z[{out['z_min']:+.2f},{out['z_max']:+.2f}]; bin/full eps {out['bin_fit_low_bias_mean']:.3f}", flush=True)
    return out


def contrast_analysis(floor=0.3):
    """Idle contrast loss per bin, per (job, rounds, patch); here per qubit contrast then mean,
    SE by bootstrap over shots within pieces (not delta method)."""
    by_job = defaultdict(list)
    for job in JOBS:
        for n in sorted({s.rounds for s in SETTINGS if s.kind == "idle"}):
            ix = next(i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.rounds == n and s.readout == "X")
            iy = next(i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.rounds == n and s.readout == "Y")
            for p in META[job]["patches"]:
                X = np.stack([D[f"idle/{job}/{k}/{p}/data"] for k in pieces_of(job, ix)]).astype(float)
                Y = np.stack([D[f"idle/{job}/{k}/{p}/data"] for k in pieces_of(job, iy)]).astype(float)
                ex, ey = 1 - 2 * X.mean((0, 1)), 1 - 2 * Y.mean((0, 1))
                keep = np.sqrt(ex ** 2 + ey ** 2) > floor
                if not keep.any():
                    continue

                def loss_in(Xb, Yb):
                    x = 1 - 2 * Xb[..., keep].mean((0, 1))
                    y = 1 - 2 * Yb[..., keep].mean((0, 1))
                    return float((1 - np.sqrt(x ** 2 + y ** 2)).mean())

                L, S = [], []
                for b in range(NB):
                    Xb, Yb = X[:, BIN == b], Y[:, BIN == b]
                    L.append(loss_in(Xb, Yb))
                    boots = []
                    for _ in range(60):
                        ix_ = RNG.integers(0, Xb.shape[1], Xb.shape[1])
                        iy_ = RNG.integers(0, Yb.shape[1], Yb.shape[1])
                        boots.append(loss_in(Xb[:, ix_], Yb[:, iy_]))
                    S.append(np.std(boots, ddof=1))
                L, S = np.array(L), np.array(S)
                m = L.mean()
                by_job[job].append((L / m, S / m))
    per = {}
    for job, rows in by_job.items():
        b, se0, red, se = wls_common(rows)
        per[job] = {"slope": b, "se": se, "chi2_red": red}
    s = np.array([per[j]["slope"] for j in per])
    e = np.array([per[j]["se"] for j in per])
    out = {"dl": dl(s, e), "positive": int((s > 0).sum()), "per_job": per,
           "z_min": float((s / e).min()), "z_max": float((s / e).max())}
    print(f"idle contrast loss (bootstrap SE) RE {100*out['dl']['re']:+.2f} [{100*out['dl']['re_ci'][0]:+.2f},{100*out['dl']['re_ci'][1]:+.2f}]"
          f" pos {out['positive']}/{len(s)} z[{out['z_min']:+.2f},{out['z_max']:+.2f}]", flush=True)
    return out


def main():
    res = {}
    res["detection_events"] = analyse(mem_series("det"), do_perm=True, name="det")
    res["det_d3"] = analyse(mem_series("det", patch="3"), name="det d3")
    res["det_d5"] = analyse(mem_series("det", patch="5"), name="det d5")
    res["det_X"] = analyse(mem_series("det", basis="X"), name="det X")
    res["det_Z"] = analyse(mem_series("det", basis="Z"), name="det Z")
    res["flags_Z"] = analyse(mem_series("flag", basis="Z"), do_perm=True, name="flags Z")
    res["fail"] = analyse(mem_series("fail"), do_perm=True, name="logical fail")
    res["idle_anc"] = analyse(idle_series("anc"), do_perm=True, name="idle anc")
    res["idle_data_X"] = analyse(idle_series("data", "X"), name="idle data 1s X")
    res["idle_data_Y"] = analyse(idle_series("data", "Y"), name="idle data 1s Y")
    res["eps"] = eps_analysis()
    res["contrast"] = contrast_analysis()
    # Stouffer z for idle Y
    pj = res["idle_data_Y"]["per_job"]
    zs = np.array([pj[j]["z"] for j in pj])
    res["idle_data_Y"]["stouffer"] = float(zs.sum() / sqrt(len(zs)))
    print("idle Y stouffer", res["idle_data_Y"]["stouffer"])
    for key in ("detection_events", "flags_Z", "fail", "idle_anc"):
        print(key)
        for j, d in res[key]["per_job"].items():
            print(f"   {j:18s} {100*d['slope']:+7.2f} ± {100*d['se']:.2f} (z {d['z']:+.2f}, chi2r {d['chi2_red']:.2f})"
                  f" curve {100*d['curve_slope']:+7.2f} ± {100*d['curve_se']:.2f}"
                  + (f" permSE {100*d['perm_se']:.2f}" if 'perm_se' in d else ""))
    (HERE / "v_heating.json").write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
