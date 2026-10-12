"""(3) Independent re-decode of every hardware shot, calibrated vs uniform weights.

Own code: own fit (weighted least squares written out, cross-checked against the
stored analysis.json eps), own rep assembly from run.json rep info, own pooling,
own paired bootstrap. Detectors come from the cache (repo model m2d of the saved shots).

    python v_redecode.py [--boot 200] [--simulate 20000]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from math import exp, sqrt

import numpy as np
from scipy.stats import binomtest, t as student_t

from vcommon import HERE, MEM, ROUNDS, cal_matching, chip, folders, meta, model, uni_matching

DEC = ("cal", "uni")


def fit(ps, N):
    """ln(1-2P) = a + b n, weights 1/var, var = 4P(1-P)/(N(1-2P)^2); P<=0 or >=0.5 dropped.
    eps = (1-e^b)/2, se = e^b/2 * se_b * sqrt(max(chi2/(k-2), 1))."""
    p = np.asarray(ps, float)
    n = np.asarray(ROUNDS, float)
    keep = (p > 0) & (p < 0.5)
    if keep.sum() < 2:
        return 0.5, float("nan"), int(keep.sum())
    p, n = p[keep], n[keep]
    var = 4 * p * (1 - p) / (N * (1 - 2 * p) ** 2)
    w = 1 / var
    y = np.log(1 - 2 * p)
    S, Sx, Sy, Sxx, Sxy = w.sum(), (w * n).sum(), (w * y).sum(), (w * n * n).sum(), (w * n * y).sum()
    D = S * Sxx - Sx * Sx
    b = (S * Sxy - Sx * Sy) / D
    a = (Sxx * Sy - Sx * Sxy) / D
    var_b = S / D
    k = len(n)
    chi2r = float((w * (y - a - b * n) ** 2).sum() / (k - 2)) if k > 2 else 1.0
    return (1 - exp(b)) / 2, exp(b) / 2 * sqrt(var_b) * sqrt(max(chi2r, 1.0)), k


def decode_folder(args):
    folder, nsim = args
    me = meta(folder)
    out = {"folder": folder, "day": me["rep"]["day"], "rep": me["rep"]["rep"], "patches": me["patches"],
           "fails": {}, "sim": {}}
    for basis in "XZ":
        for n in ROUNDS:
            idx = MEM[(basis, n)]
            m = model(folder, idx)
            det, obs = chip(folder, idx)
            cal = cal_matching(m)
            uni = uni_matching(cal)
            fc = cal.decode_batch(det).astype(bool) != obs
            fu = uni.decode_batch(det).astype(bool) != obs
            saved = np.array(me["settings"][str(idx)]["saved_logical_error"])
            assert np.array_equal(fc.mean(0), saved), (folder, idx, fc.mean(0), saved)
            out["fails"][(basis, n)] = (fc, fu)
            if nsim:
                sd, so = m.compile_detector_sampler(seed=1000 + 17 * idx).sample(nsim, separate_observables=True)
                out["sim"][(basis, n)] = ((cal.decode_batch(sd).astype(bool) != so).mean(0),
                                          (uni.decode_batch(sd).astype(bool) != so).mean(0))
    return out


def eps_table(fails_by_folder, N_of, which="fails", resample=None):
    """{(folder, basis, patch, dec): (eps, se)} from per-setting failure arrays."""
    res = {}
    for f, info in fails_by_folder.items():
        for basis in "XZ":
            for k, patch in enumerate(info["patches"]):
                for di, dec in enumerate(DEC):
                    ps = []
                    for n in ROUNDS:
                        if which == "fails":
                            arr = info["fails"][(basis, n)][di][:, k]
                            if resample is not None:
                                arr = arr[resample[(f, basis, n)]]
                            ps.append(arr.mean())
                        else:
                            ps.append(info["sim"][(basis, n)][di][k])
                    e, s, _ = fit(ps, N_of(f, which))
                    res[(f, basis, patch, dec)] = (e, s)
    return res


def reps_of(infos, eps):
    reps = defaultdict(dict)
    for f, info in infos.items():
        for basis in "XZ":
            for patch in info["patches"]:
                for dec in DEC:
                    reps[(info["day"], info["rep"])][(basis, patch, dec)] = eps[(f, basis, patch, dec)]
    rows = []
    for (day, rep), e in sorted(reps.items()):
        row = {"day": day, "rep": rep}
        for basis in "XZ":
            for dec in DEC:
                (e3, s3), (e5, s5) = e[(basis, "3", dec)], e[(basis, "5", dec)]
                lam = e3 / e5
                row[f"{basis} {dec}"] = {"eps3": [e3, s3], "eps5": [e5, s5],
                                         "lambda": [lam, lam * sqrt((s3 / e3) ** 2 + (s5 / e5) ** 2)]}
        rows.append(row)
    return rows


def pool(v, s):
    v, s = np.asarray(v), np.asarray(s)
    w = 1 / s**2
    m = float((w * v).sum() / w.sum())
    chi2nu = float((w * (v - m) ** 2).sum() / (len(v) - 1))
    se = sqrt(max(chi2nu, 1.0) / w.sum())
    t = float(student_t.ppf(0.95, len(v) - 1))
    verdict = "d = 5 better" if m - t * se > 1 else "d = 5 worse" if m + t * se < 1 else "undecided"
    return {"lambda": m, "se": se, "chi2nu": chi2nu, "t": t, "verdict": verdict}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--simulate", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    infos = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for info in ex.map(decode_folder, [(f, a.simulate) for f in folders()]):
            infos[info["folder"]] = info
            print(info["folder"], "decoded", flush=True)
    N = {f: infos[f]["fails"][("X", 1)][0].shape[0] for f in infos}

    def N_of(f, which):
        return N[f] if which == "fails" else a.simulate

    # cross-check own fit against stored analysis eps
    maxdiff = 0.0
    for f in infos:
        an = json.loads((HERE.parents[3] / "runs" / f / "analysis.json").read_text())
        for basis in "XZ":
            for k, patch in enumerate(infos[f]["patches"]):
                ps = [infos[f]["fails"][(basis, n)][0][:, k].mean() for n in ROUNDS]
                e, s, _ = fit(ps, N[f])
                e0, s0 = an["observed"]["fits"][basis]["per_round"][patch]
                maxdiff = max(maxdiff, abs(e - e0) / e0, abs(s - s0) / s0)
    print("max relative diff of own fit vs analysis.json eps/se:", maxdiff)

    eps = eps_table(infos, N_of)
    rows = reps_of(infos, eps)
    combined = json.loads((HERE.parents[3] / "runs" / "combined.json").read_text())
    for r, c in zip(rows, combined["reps"]):
        assert (r["day"], r["rep"]) == (c["day"], c["rep"])
        assert abs(r["X cal"]["lambda"][0] - c["tests"]["X lambda"]["observed"]) < 1e-9
        assert abs(r["X cal"]["lambda"][1] - c["tests"]["X lambda"]["observed_uncertainty"]) < 1e-6
    out = {"N": N, "reps": rows, "pooled": {}, "mean_eps": {}, "mcnemar": {}, "P": {}, "P_sim": {}}
    dropped = defaultdict(int)
    for f, info in infos.items():
        for (basis, n), (fc, fu) in info["fails"].items():
            out["P"].setdefault(f, {})[f"{basis}{n}"] = {"cal": fc.mean(0).tolist(), "uni": fu.mean(0).tolist()}
            for k, patch in enumerate(info["patches"]):
                for dec, arr in (("cal", fc), ("uni", fu)):
                    if arr[:, k].mean() >= 0.5:
                        dropped[f"{basis} d{patch} {dec}"] += 1
            if a.simulate:
                sc, su = info["sim"][(basis, n)]
                out["P_sim"].setdefault(f, {})[f"{basis}{n}"] = {"cal": list(map(float, sc)), "uni": list(map(float, su))}
    out["points_dropped_P_ge_half"] = dict(dropped)
    print("points with P>=0.5:", dict(dropped))
    for basis in "XZ":
        for dec in DEC:
            v = [r[f"{basis} {dec}"]["lambda"][0] for r in rows]
            s = [r[f"{basis} {dec}"]["lambda"][1] for r in rows]
            out["pooled"][f"{basis} {dec}"] = pool(v, s)
            for q in ("eps3", "eps5"):
                x = np.array([r[f"{basis} {dec}"][q][0] for r in rows])
                out["mean_eps"][f"{basis} {q} {dec}"] = [float(x.mean()), float(x.std(ddof=1) / sqrt(len(x)))]
    # McNemar
    for basis in "XZ":
        for patch in "35":
            co = uo = both = tot = 0
            for f, info in infos.items():
                if patch not in info["patches"]:
                    continue
                k = info["patches"].index(patch)
                for n in ROUNDS:
                    fc, fu = info["fails"][(basis, n)]
                    co += int((fc[:, k] & ~fu[:, k]).sum())
                    uo += int((fu[:, k] & ~fc[:, k]).sum())
                    both += int((fu[:, k] & fc[:, k]).sum())
                    tot += fc.shape[0]
            out["mcnemar"][f"{basis} d{patch}"] = {"cal_only": co, "uni_only": uo, "both": both, "shots": tot,
                                                   "p": float(binomtest(co, co + uo, 0.5).pvalue)}
    # paired bootstrap of pooled difference
    if a.boot:
        rng = np.random.default_rng(20261010)
        diffs = {b: [] for b in "XZ"}
        diffs_rw = {b: [] for b in "XZ"}
        lam_rep = {b: [] for b in "XZ"}
        for _ in range(a.boot):
            rs = {}
            for f in infos:
                for basis in "XZ":
                    for n in ROUNDS:
                        rs[(f, basis, n)] = rng.integers(0, N[f], N[f])
            eb = eps_table(infos, N_of, resample=rs)
            rb = reps_of(infos, eb)
            for basis in "XZ":
                vals, vals_rw = {}, {}
                for dec in DEC:
                    lv = np.array([r[f"{basis} {dec}"]["lambda"][0] for r in rb])
                    ls = np.array([r[f"{basis} {dec}"]["lambda"][1] for r in rb])
                    w0 = 1 / np.array([r[f"{basis} {dec}"]["lambda"][1] for r in rows]) ** 2
                    vals[dec] = float((w0 * lv).sum() / w0.sum())
                    w1 = 1 / ls**2
                    vals_rw[dec] = float((w1 * lv).sum() / w1.sum())
                diffs[basis].append(vals["uni"] - vals["cal"])
                diffs_rw[basis].append(vals_rw["uni"] - vals_rw["cal"])
                lam_rep[basis].append([r[f"{basis} uni"]["lambda"][0] - r[f"{basis} cal"]["lambda"][0] for r in rb])
        for basis in "XZ":
            d = np.array(diffs[basis])
            drw = np.array(diffs_rw[basis])
            out["pooled"][f"{basis} diff"] = {
                "uni_minus_cal": out["pooled"][f"{basis} uni"]["lambda"] - out["pooled"][f"{basis} cal"]["lambda"],
                "boot_sd_fixed_weights": float(np.nanstd(d, ddof=1)),
                "boot_sd_reweighted": float(np.nanstd(drw, ddof=1)),
                "boot_mean_fixed_weights": float(np.nanmean(d)),
                "B": a.boot,
                "per_rep_diff_sd": np.nanstd(np.array(lam_rep[basis]), axis=0, ddof=1).tolist(),
            }
    if a.simulate:
        es = eps_table(infos, N_of, which="sim")
        rs = reps_of(infos, es)
        out["sim"] = {"shots": a.simulate, "pooled": {}, "reps": rs}
        for basis in "XZ":
            for dec in DEC:
                v = [r[f"{basis} {dec}"]["lambda"][0] for r in rs]
                s = [r[f"{basis} {dec}"]["lambda"][1] for r in rs]
                out["sim"]["pooled"][f"{basis} {dec}"] = pool(v, s)
                out["sim"]["pooled"][f"{basis} {dec}"]["unweighted_mean"] = float(np.mean(v))
    name = "v_redecode.json" if not a.simulate else f"v_redecode_sim{a.simulate}.json"
    (HERE / name).write_text(json.dumps(out, indent=1, default=str))
    for r in rows:
        print(r["day"], r["rep"], " ".join(
            f"{b}: cal {r[f'{b} cal']['lambda'][0]:.4f}+-{r[f'{b} cal']['lambda'][1]:.4f} uni {r[f'{b} uni']['lambda'][0]:.4f}+-{r[f'{b} uni']['lambda'][1]:.4f}"
            for b in "XZ"))
    print(json.dumps(out["pooled"], indent=1))
    print(json.dumps(out["mean_eps"], indent=1))
    print(json.dumps(out["mcnemar"], indent=1))
    if a.simulate:
        print(json.dumps(out["sim"]["pooled"], indent=1))


if __name__ == "__main__":
    main()
