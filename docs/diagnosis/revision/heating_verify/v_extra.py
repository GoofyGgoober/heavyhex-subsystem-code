"""Extra checks for the heating verification (reads v_per_shot.npz, v_meta.json, v_heating.json).

1. Hartung-Knapp (HKSJ) random-effects CIs from my per-job slopes (DL with k=15 is anti-conservative).
2. Their exact eps-per-bin recipe re-implemented (normalize by mean over bins; one error per series =
   mean of bin fit errors; WLS common slope per job with chi2 inflation), to check -1.4 % and 7/15.
3. Per-qubit idle-ancilla slopes (540 qubit-jobs?), counts of |z|>3, median slope, named examples.
4. Pooled 10-bin curves of the memory metrics: is any bin significantly above 1 ("rose at any point")?
5. H2 (time = piece position k) slope for detection events.
6. Idle contrast loss with delta-method errors (their recipe) to check -0.48 % and 5/15.
7. Adjacent-shot correlation under H1/H2 orderings, scored independently.
"""

from __future__ import annotations

import json
from collections import defaultdict
from math import sqrt

import numpy as np
from scipy import stats

from heavyhex.experiment.circuits import ROUNDS, SETTINGS
from heavyhex.simulation.noisy import fit_per_round

import v_heating as V  # the verification's own module (same folder)

HERE = V.HERE
D, META, JOBS, NB, N, BIN, TB = V.D, V.META, V.JOBS, V.NB, V.N, V.BIN, V.TB
RES = json.loads((HERE / "v_heating.json").read_text())
out = {}


def hksj(s, e):
    s, e = np.asarray(s), np.asarray(e)
    k = len(s)
    d = V.dl(s, e)
    w = 1 / (e ** 2 + d["tau"] ** 2)
    mu = (w * s).sum() / w.sum()
    q = (w * (s - mu) ** 2).sum() / (k - 1)
    se = sqrt(q / w.sum())
    t = stats.t.ppf(0.975, k - 1)
    return {"mu": mu, "se": se, "ci": [mu - t * se, mu + t * se]}


print("1. HKSJ intervals (per-job slopes and SEs from v_heating (a))")
out["hksj"] = {}
for key in ("detection_events", "det_d3", "det_d5", "flags_Z", "fail", "idle_anc", "idle_data_Y"):
    pj = RES[key]["per_job"]
    s = [pj[j]["slope"] for j in pj]
    e = [pj[j]["se"] for j in pj]
    h = hksj(s, e)
    out["hksj"][key] = h
    print(f"   {key:20s} HKSJ {100*h['mu']:+.2f} [{100*h['ci'][0]:+.2f}, {100*h['ci'][1]:+.2f}] %")


print("2. eps per bin, their recipe")
by_job = defaultdict(list)
series_slopes = []
for job in JOBS:
    for basis in "XZ":
        idx = [i for i, s in enumerate(SETTINGS) if s.kind == "memory" and s.basis == basis]
        for p in META[job]["patches"]:
            F = [np.stack([D[f"mem/{job}/{k}/{p}/fail"] for k in V.pieces_of(job, i)]).astype(float) for i in idx]
            eps, unc = [], []
            for b in range(NB):
                fit = fit_per_round(ROUNDS, [f[:, BIN == b].mean() for f in F], F[0][:, BIN == b].size)
                eps.append(fit.per_round)
                unc.append(fit.uncertainty)
            eps, unc = np.array(eps), np.array(unc)
            ok = np.isfinite(unc) & (unc > 0)
            if ok.sum() < 3:
                continue
            t = (np.arange(NB) + 0.5) / NB
            norm = eps[ok].mean()
            y = eps[ok] / norm
            se = np.full(ok.sum(), unc[ok].mean() / norm)
            by_job[job].append((t[ok], y, se))
            w = 1 / se ** 2
            tb = (w * t[ok]).sum() / w.sum()
            series_slopes.append((w * (t[ok] - tb) * (y - (w * y).sum() / w.sum())).sum() / (w * (t[ok] - tb) ** 2).sum())


def common(rows):
    num = den = 0.0
    keep = []
    for t, y, s in rows:
        w = 1 / s ** 2
        tb, yb = (w * t).sum() / w.sum(), (w * y).sum() / w.sum()
        num += (w * (t - tb) * (y - yb)).sum()
        den += (w * (t - tb) ** 2).sum()
        keep.append((t, y, w, tb, yb))
    b = num / den
    chi2 = sum((w * (y - yb - b * (t - tb)) ** 2).sum() for t, y, w, tb, yb in keep)
    dof = sum(len(t) - 1 for t, *_ in keep) - 1
    return b, sqrt(1 / den) * sqrt(max(chi2 / dof, 1))


pj = {j: common(r) for j, r in by_job.items()}
s = np.array([v[0] for v in pj.values()])
e = np.array([v[1] for v in pj.values()])
d = V.dl(s, e)
h = hksj(s, e)
out["eps_their_recipe"] = {"dl": d, "hksj": h, "positive_jobs": int((s > 0).sum()), "series": len(series_slopes),
                           "series_positive": int((np.array(series_slopes) > 0).sum()),
                           "z": (s / e).tolist()}
print(f"   RE {100*d['re']:+.2f} [{100*d['re_ci'][0]:+.2f}, {100*d['re_ci'][1]:+.2f}]; HKSJ [{100*h['ci'][0]:+.2f}, {100*h['ci'][1]:+.2f}];"
      f" positive jobs {(s>0).sum()}/15; series positive {out['eps_their_recipe']['series_positive']}/{len(series_slopes)}; z [{(s/e).min():+.2f},{(s/e).max():+.2f}]")


print("3. per-qubit idle ancilla")
rows = []
for job in JOBS:
    idle_ks = [int(k) for k, p in META[job]["pieces"].items() if p["kind"] == "idle"]
    qubits = sorted({int(key.split("/")[3]) for key in D.files if key.startswith(f"q/{job}/{idle_ks[0]}/")})
    # data qubits: those in d registers -> exclude. Identify by memory? Use: a qubit is data if its idle reads
    # are 1 at ~50% (X/Y readout). Better: from circuits; here use the stored per-piece readout counts:
    for q in qubits:
        series = []
        for i, s_ in enumerate(SETTINGS):
            if s_.kind != "idle":
                continue
            ks = V.pieces_of(job, i)
            Vq = np.stack([D[f"q/{job}/{k}/{q}"] for k in ks]).astype(float)
            if Vq.mean() > 0.3:  # data qubit (X/Y readout of |+>-ish), not an ancilla
                break
            r = V.bins_of(Vq)
            if r is not None:
                series.append(r)
        else:
            if series:
                b, se0, red, se = V.wls_common(series)
                curve = np.mean([r[0] for r in series], axis=0)
                rows.append({"job": job, "qubit": q, "slope": b, "se": se, "z": b / se, "curve": curve.tolist()})
z = np.array([r["z"] for r in rows])
out["per_qubit_idle_anc"] = {"n": len(rows), "positive": int((z > 0).sum()), "z_gt3": int((z > 3).sum()),
                             "z_lt_m3": int((z < -3).sum()), "expected_each": float(len(rows) * stats.norm.sf(3)),
                             "median_slope": float(np.median([r["slope"] for r in rows])),
                             "largest": sorted(rows, key=lambda r: -abs(r["z"]))[:14]}
print(f"   {len(rows)} qubit-jobs, positive {(z>0).sum()}, z>3 {(z>3).sum()}, z<-3 {(z<-3).sum()}, "
      f"expected {len(rows)*stats.norm.sf(3):.2f} each, median slope {100*np.median([r['slope'] for r in rows]):+.2f}%")
for r in out["per_qubit_idle_anc"]["largest"]:
    print(f"     {r['job']} q{r['qubit']}: {100*r['slope']:+.1f} ± {100*r['se']:.1f} (z {r['z']:+.2f}) curve {np.round(r['curve'], 2).tolist()}")


print("4. pooled bin curves")
out["pooled_bins"] = {}
for key, ser in (("det", V.mem_series("det")), ("flag", V.mem_series("flag", basis="Z")), ("fail", V.mem_series("fail"))):
    Y, W = [], []
    for job, lab, Vv in ser:
        r = V.bins_of(Vv)
        if r is None:
            continue
        Y.append(r[0])
        W.append(1 / r[1] ** 2)
    Y, W = np.array(Y), np.array(W)
    m = (W * Y).sum(0) / W.sum(0)
    se = 1 / np.sqrt(W.sum(0))
    red = (W * (Y - m) ** 2).sum(0) / (len(Y) - 1)
    se = se * np.sqrt(np.maximum(red, 1))
    zb = (m - 1) / se
    out["pooled_bins"][key] = {"y": m.tolist(), "se": se.tolist(), "z": zb.tolist()}
    print(f"   {key}: y-1 (%) {np.round(100*(m-1), 2).tolist()}  max z {zb.max():+.2f} min z {zb.min():+.2f}")


print("5. H2: time = piece position")
def h2(ser, total=68):
    by = defaultdict(list)
    for job, (i, p), Vv in ser:
        ks = V.pieces_of(job, i)
        m = Vv.mean()
        if m <= 0:
            continue
        resid = Vv - Vv.mean(1, keepdims=True)
        var = (resid ** 2).sum() / (Vv.size - len(ks))
        t = (np.array(ks) + 0.5) / total
        by[job].append((t, Vv.mean(1) / m, np.full(len(ks), sqrt(var / N) / m)))
    pj = {j: common(r) for j, r in by.items()}
    s = np.array([v[0] for v in pj.values()])
    e = np.array([v[1] for v in pj.values()])
    return V.dl(s, e), int((s > 0).sum())
d2, pos = h2(V.mem_series("det"))
out["h2_det"] = {"dl": d2, "positive": pos}
print(f"   det under H2: RE {100*d2['re']:+.2f} [{100*d2['re_ci'][0]:+.2f}, {100*d2['re_ci'][1]:+.2f}], FE {100*d2['fe']:+.2f} ± {100*d2['fe_se']:.2f}, pos {pos}/15")


print("6. idle contrast loss, delta method (their recipe)")
by = defaultdict(list)
for job in JOBS:
    for n in sorted({s.rounds for s in SETTINGS if s.kind == "idle"}):
        ix = next(i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.rounds == n and s.readout == "X")
        iy = next(i for i, s in enumerate(SETTINGS) if s.kind == "idle" and s.rounds == n and s.readout == "Y")
        for p in META[job]["patches"]:
            X = np.stack([D[f"idle/{job}/{k}/{p}/data"] for k in V.pieces_of(job, ix)]).astype(float)
            Y = np.stack([D[f"idle/{job}/{k}/{p}/data"] for k in V.pieces_of(job, iy)]).astype(float)
            ex, ey = 1 - 2 * X.mean((0, 1)), 1 - 2 * Y.mean((0, 1))
            keep = np.sqrt(ex ** 2 + ey ** 2) > 0.3
            if not keep.any():
                continue
            L, S = [], []
            for b in range(NB):
                x = 1 - 2 * X[:, BIN == b][..., keep].mean((0, 1))
                y = 1 - 2 * Y[:, BIN == b][..., keep].mean((0, 1))
                nx = X[:, BIN == b].shape[0] * (BIN == b).sum()
                ny = Y[:, BIN == b].shape[0] * (BIN == b).sum()
                c = np.sqrt(x ** 2 + y ** 2)
                vc = (x ** 2 * (1 - x ** 2) / nx + y ** 2 * (1 - y ** 2) / ny) / np.maximum(c, 1e-9) ** 2
                L.append((1 - c).mean())
                S.append(np.sqrt(vc.sum()) / keep.sum())
            L, S = np.array(L), np.array(S)
            by[job].append(((np.arange(NB) + 0.5) / NB, L / L.mean(), S / L.mean()))
pj = {j: common(r) for j, r in by.items()}
s = np.array([v[0] for v in pj.values()])
e = np.array([v[1] for v in pj.values()])
d6 = V.dl(s, e)
out["contrast_delta"] = {"dl": d6, "positive": int((s > 0).sum())}
print(f"   RE {100*d6['re']:+.2f} [{100*d6['re_ci'][0]:+.2f}, {100*d6['re_ci'][1]:+.2f}], pos {(s>0).sum()}/15")


print("7. adjacent-shot correlation")
acc = defaultdict(lambda: [0.0, 0])
for job in JOBS:
    K = len(META[job]["pieces"])
    Z = np.zeros((K, N))
    for k in range(K):
        info = META[job]["pieces"][str(k)]
        v = np.zeros(N)
        for p in META[job]["patches"]:
            if info["kind"] == "memory":
                v += D[f"mem/{job}/{k}/{p}/det"] + D[f"mem/{job}/{k}/{p}/flag"]
            else:
                v += D[f"idle/{job}/{k}/{p}/anc"]
        Z[k] = (v - v.mean()) / v.std()
    h1 = Z.T.reshape(-1)
    h2_ = Z.reshape(-1)
    for lag in (1, 2, 3, 5, 10, 68):
        for nm, fl in (("H1", h1), ("H2", h2_)):
            pr = fl[:-lag] * fl[lag:]
            acc[f"{nm} lag {lag}"][0] += pr.sum()
            acc[f"{nm} lag {lag}"][1] += len(pr)
out["adjacent"] = {k: {"corr": a / n, "se": 1 / sqrt(n)} for k, (a, n) in acc.items()}
for k, v in out["adjacent"].items():
    print(f"   {k}: {v['corr']:+.4f} ± {v['se']:.4f}")

(HERE / "v_extra.json").write_text(json.dumps(out, indent=1, default=float))
