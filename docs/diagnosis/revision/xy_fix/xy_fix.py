"""The idle test's f with the correct variance of ln r for separately sampled <X> and <Y>.

Post hoc, exploratory; reads the committed run folders, changes nothing in src/ or runs/,
uses no QPU. Stage 1 caches, per job and data qubit, <X>, <Y>, the shot counts and the
exact model references ln x0, ln x1 (src/heavyhex/experiment/run.py, _idle_dephasing).
Stage 2 refits f with

  old (run.py:1059)   Var(ln r) = (1 - r^2) / (N r^2),   N = (N_X + N_Y) / 2
  new (delta method)  r2 = (1 + 1/N_X) x^2 + (1 + 1/N_Y) y^2 - 1/N_X - 1/N_Y   (as run.py:947)
                      Var(r2) = 4 [(1+1/N_X)^2 x^2 (1-x^2)/N_X + (1+1/N_Y)^2 y^2 (1-y^2)/N_Y]
                      Var(ln r) = Var(r2) / (4 r2^2)

and reports every downstream idle-test number the paper uses.
"""

from __future__ import annotations

import json
import sys
from math import sqrt
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "src"))

from heavyhex.experiment import run as R  # noqa: E402
from heavyhex.experiment.circuits import SETTINGS  # noqa: E402
from heavyhex.patches.placement import Calibration  # noqa: E402

CACHE = HERE / "idle_inputs.json"


def job_folders() -> list[Path]:
    out = []
    for folder in sorted((REPO / "runs").glob("ibm_fez-2026-10-*-r*")):
        info = json.loads((folder / "run.json").read_text())
        if (folder / "analysis.json").exists() and info.get("rep") and not info.get("offline"):
            out.append(folder)
    return out


def stage1() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text())
    found = {}
    for folder in job_folders():
        info = json.loads((folder / "run.json").read_text())
        calibration = Calibration.load(folder / "calibration.json")
        circuits = R.load_circuits(folder)
        shots = R.load_shots(folder)
        idle = [i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.readout == "X"]
        in_y = {s.rounds: i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.readout == "Y"}
        rounds = [SETTINGS[i].rounds for i in idle]
        xs = [R.x_values(shots[i]) for i in idle]
        ys = [R.x_values(shots[in_y[n]]) for n in rounds]
        ends = [[R._exact_x(circuits[i], calibration, info["dt"], f) for i in idle] for f in R.ENDS]
        job = {"rounds": rounds, "day": info["rep"]["day"], "rep": info["rep"]["rep"],
               "chips": info["chips"], "qubits": {}}
        for d in xs[0]:
            for q in range(int(d) ** 2):
                job["qubits"][f"{d}/{q}"] = {
                    "x": [v[d][q] for v in xs],
                    "y": [v[d][q] for v in ys],
                    "nx": [len(shots[i][f"d{d}"]) for i in idle],
                    "ny": [len(shots[in_y[n]][f"d{d}"]) for n in rounds],
                    "log_x0": [float(np.log(e[d][q])) for e in ends[0]],
                    "log_x1": [float(np.log(e[d][q])) for e in ends[1]],
                }
        found[folder.name] = job
        print("cached", folder.name, flush=True)
    CACHE.write_text(json.dumps(found))
    return found


def fit(seen, log_x0, log_x1, var_log):
    """run.py _fit_share with the weight 1/var_log passed in (identical otherwise)."""
    keep = seen > 0.05
    extra = (log_x0 - np.log(np.where(keep, seen, 1)))[keep]
    full = (log_x0 - log_x1)[keep]
    weight = (1 / var_log)[keep]
    nan = (float("nan"), float("nan"))
    if not keep.any():
        return nan
    if keep.sum() == 1:
        if full[0] <= 0:
            return nan
        share, sigma = extra[0] / full[0], 1 / (full[0] * sqrt(weight[0]))
    else:
        w, wx, wy = weight.sum(), (weight * full).sum(), (weight * extra).sum()
        spread = w * (weight * full**2).sum() - wx**2
        if spread <= 0:
            return nan
        share = (w * (weight * full * extra).sum() - wx * wy) / spread
        sigma = sqrt(w / spread)
    return (float(share), float(sigma)) if sigma <= R.LOOSEST_F else nan


def variances(q):
    x, y = np.array(q["x"]), np.array(q["y"])
    nx, ny = np.array(q["nx"], float), np.array(q["ny"], float)
    r2 = x**2 + y**2 - (1 - x**2) / nx - (1 - y**2) / ny
    r = np.sqrt(np.maximum(r2, 0))
    shots = (nx + ny) / 2
    old = (1 - np.minimum(r, 0.999) ** 2) / (shots * r**2)  # exactly 1/weight of run.py:1059
    var_r2 = 4 * ((1 + 1 / nx) ** 2 * x**2 * (1 - x**2) / nx + (1 + 1 / ny) ** 2 * y**2 * (1 - y**2) / ny)
    new = var_r2 / (4 * np.maximum(r2, 1e-300) ** 2)
    return r, old, new


def summary(shares, sigmas):
    s, u = np.array(shares), np.array(sigmas)
    ok = np.isfinite(s) & np.isfinite(u)
    s, u = s[ok], u[ok]
    uni = R._uniformity(s, u)
    return {
        "n": int(ok.sum()),
        "median": float(np.median(s)),
        "median_uncertainty": float(1.25 * np.std(s) / sqrt(len(s))),
        "chi2_per_dof": uni["chi2_per_dof"],
        "weighted_mean": uni["weighted_mean"],
    }


def main() -> None:
    from scipy.stats import spearmanr

    data = stage1()
    ratios, rows, out = [], [], {}
    per_old, per_new = [], []
    for name, job in data.items():
        a = json.loads((REPO / "runs" / name / "analysis.json").read_text())
        so, uo, sn, un = [], [], [], []
        for key, q in job["qubits"].items():
            d, k = key.split("/")
            r, old, new = variances(q)
            keep = r > 0.05
            ratios += list(np.sqrt(new[keep] / old[keep]))
            for n, rr, o, nw, x, y in zip(job["rounds"], r, old, new, q["x"], q["y"]):
                if rr > 0.05:
                    rows.append((sqrt(nw / o), name, key, job["chips"][d][int(k)], n, x, y, sqrt(o), sqrt(nw)))
            l0, l1 = np.array(q["log_x0"]), np.array(q["log_x1"])
            f_o, s_o = fit(r, l0, l1, old)
            f_n, s_n = fit(r, l0, l1, new)
            # reproduce the prespecified numbers exactly
            ref_f, _ref_s = a["idle_test"]["per_qubit"][d][int(k)], a["idle_test"]["uncertainty"][d][int(k)]
            assert (np.isnan(f_o) and (ref_f is None or not np.isfinite(ref_f))) or abs(f_o - ref_f) < 1e-9, (name, key, f_o, ref_f)
            so.append(f_o); uo.append(s_o); sn.append(f_n); un.append(s_n)
            per_old.append((name, key, f_o, s_o)); per_new.append((name, key, f_n, s_n))
        o, n = summary(so, uo), summary(sn, un)
        assert abs(o["median"] - a["idle_test"]["median"]) < 1e-12
        assert abs(o["median_uncertainty"] - a["idle_test"]["median_uncertainty"]) < 1e-12
        out[name] = {"old": o, "new": n}
    ratios = np.array(ratios)
    print(f"points kept (r>0.05): {len(ratios)}")
    print(f"SE(ln r) new/old: median {np.median(ratios):.4f}  95th {np.percentile(ratios, 95):.4f}  "
          f"max {ratios.max():.4f}  min {ratios.min():.4f}  >1.10: {(ratios > 1.10).sum()}")
    rows.sort(reverse=True)
    print("largest:", rows[0])
    ex = [r for r in rows if r[1] == "ibm_fez-2026-10-06-r2-d5" and r[3] == 70 and r[4] == 1]
    print("reviewer example (Q70, d5, day3 rep2, n=1):", ex)

    fo = np.array([p[2] for p in per_old]); fn = np.array([p[2] for p in per_new])
    uo = np.array([p[3] for p in per_old]); un = np.array([p[3] for p in per_new])
    both = np.isfinite(fo) & np.isfinite(fn)
    print(f"\nper-qubit: measured old {np.isfinite(fo).sum()}  new {np.isfinite(fn).sum()} of {len(fo)}")
    lost = [(p[0], p[1], round(p[2], 3), round(p[3], 3), round(q[3], 3) if np.isfinite(q[3]) else None)
            for p, q in zip(per_old, per_new) if np.isfinite(p[2]) and not np.isfinite(q[2])]
    gained = [(p[0], p[1]) for p, q in zip(per_old, per_new) if not np.isfinite(p[2]) and np.isfinite(q[2])]
    print("excluded only under new variance:", lost)
    print("included only under new variance:", gained)
    df = fn[both] - fo[both]
    print(f"|Δf| per qubit: median {np.median(abs(df)):.4f}  95th {np.percentile(abs(df), 95):.4f}  max {abs(df).max():.4f}")
    zr = abs(df) / uo[both]
    print(f"|Δf|/σ_old: median {np.median(zr):.3f}  max {zr.max():.3f}")
    sr = un[both] / uo[both]
    print(f"σ_f new/old: median {np.median(sr):.4f}  95th {np.percentile(sr, 95):.4f}  max {sr.max():.4f}  min {sr.min():.4f}")
    for lab, f_, u_ in (("old", fo, uo), ("new", fn, un)):
        m = np.isfinite(f_)
        q1, q3 = np.percentile(f_[m], [25, 75])
        print(f"{lab}: negatives {int((f_[m] < 0).sum())}  IQR {q1:.2f}-{q3:.2f}  "
              f"Spearman(f, σ) {spearmanr(f_[m], u_[m]).statistic:.3f}")
    print("\njob                         median old -> new (Δ)        SE old -> new    chi2/nu old -> new   wmean old -> new   n")
    for name, v in out.items():
        o, n = v["old"], v["new"]
        print(f"{name:27s} {o['median']:.4f} -> {n['median']:.4f} ({n['median'] - o['median']:+.4f})  "
              f"{o['median_uncertainty']:.4f} -> {n['median_uncertainty']:.4f}  "
              f"{o['chi2_per_dof']:5.1f} -> {n['chi2_per_dof']:5.1f}  "
              f"{o['weighted_mean']:.3f} -> {n['weighted_mean']:.3f}  {o['n']}->{n['n']}")
    for lab in ("old", "new"):
        meds = [v[lab]["median"] for v in out.values()]
        c = [v[lab]["chi2_per_dof"] for v in out.values()]
        w = [v[lab]["weighted_mean"] for v in out.values()]
        print(f"{lab}: job medians {min(meds):.2f}-{max(meds):.2f}, median {np.median(meds):.2f}; "
              f"chi2/nu {min(c):.0f}-{max(c):.0f}; weighted mean {min(w):.2f}-{max(w):.2f}")
    (HERE / "xy_fix.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
