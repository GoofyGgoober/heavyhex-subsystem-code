"""Is shot index j within a piece the time order, with all pieces interleaved (H1)?
Or are pieces run one after another in job order k (H2)?  Independent tests.

Uses q/<job>/<k>/<qubit> arrays (per-shot fraction of that physical qubit's readouts
that read 1) from v_per_shot.npz.

Test A (q51, 2026-10-06-r2-d5): per-piece changepoint (least-squares mean shift) found
  in each piece independently -> do the 68 pieces' change points coincide?
  And: same-setting pieces in block 0 vs block 1 (under H2 block 0 runs entirely before
  block 1, so a mid-job switch makes them differ; under H1 they are identical mixtures).
Test B (all jobs, all qubits): for every (job, qubit), standardized 25-bin profiles of each
  piece; mean pairwise correlation among pieces of that qubit in that job, vs a null of
  independent circular shifts. Under H1 slow per-qubit fluctuations make it > 0.
Test C (H2 signature): per (job, qubit), regress piece means on piece position k with
  per-setting intercepts (settings appear in both blocks): under H2 a drifting qubit
  shows a trend in k; under H1 nothing.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
D = np.load(HERE / "v_per_shot.npz")
META = json.loads((HERE / "v_meta.json").read_text())
N = 625
RNG = np.random.default_rng(99)


def qarrays(job):
    out = defaultdict(dict)  # qubit -> k -> array
    for key in D.files:
        if key.startswith(f"q/{job}/"):
            _, _, k, q = key.split("/")
            out[int(q)][int(k)] = D[key].astype(float)
    return out


def changepoint(v, lo=20):
    c = np.cumsum(v)
    tot = c[-1]
    n = len(v)
    j = np.arange(lo, n - lo)
    left = c[j - 1] / j
    right = (tot - c[j - 1]) / (n - j)
    gain = j * (n - j) / n * (left - right) ** 2
    b = int(np.argmax(gain))
    return int(j[b]), float(left[b]), float(right[b])


def test_a(job="2026-10-06-r2-d5", qubit=51):
    Q = qarrays(job)[qubit]
    rows = []
    for k in sorted(Q):
        j0, a, b = changepoint(Q[k])
        rows.append((k, META[job]["pieces"][str(k)]["i"], META[job]["pieces"][str(k)]["block"], j0, a, b))
    j0s = np.array([r[3] for r in rows])
    pooled = np.mean([Q[k] for k in Q], axis=0)
    jp, a, b = changepoint(pooled)
    # block comparison for the same setting
    by_set = defaultdict(lambda: {0: [], 1: []})
    for k in Q:
        p = META[job]["pieces"][str(k)]
        by_set[p["i"]][p["block"]].append(Q[k].mean())
    _diffs = [np.mean(v[1]) - np.mean(v[0]) for v in by_set.values() if v[0] and v[1]]
    rel = [ (np.mean(v[1]) - np.mean(v[0])) / np.mean(v[0] + v[1]) for v in by_set.values() if v[0] and v[1]]
    # piece mean before/after pooled j0 per piece
    lower_after = sum(Q[k][jp:].mean() < Q[k][:jp].mean() for k in Q)
    # under H2, piece position of the switch would be about 68 * jp/625
    out = {
        "job": job, "qubit": qubit, "pieces": len(Q),
        "pooled_changepoint": jp, "rate_before": a, "rate_after": b,
        "per_piece_changepoint_median": float(np.median(j0s)),
        "per_piece_changepoint_within_25_of_pooled": int((np.abs(j0s - jp) <= 25).sum()),
        "per_piece_changepoint_iqr": [float(np.percentile(j0s, 25)), float(np.percentile(j0s, 75))],
        "pieces_lower_after_pooled_cp": int(lower_after),
        "block1_minus_block0_mean_rel": float(np.mean(rel)), "block1_minus_block0_sd_rel": float(np.std(rel, ddof=1)),
        "block_settings": len(rel),
        "per_piece": rows,
    }
    print(f"A: {job} q{qubit}: pooled cp {jp} ({a:.3f}->{b:.3f}); per-piece cp median {np.median(j0s):.0f},"
          f" within ±25 of pooled: {out['per_piece_changepoint_within_25_of_pooled']}/{len(Q)}; lower after: {lower_after}/{len(Q)};"
          f" block1-block0 rel {np.mean(rel):+.3f} ± {np.std(rel, ddof=1)/np.sqrt(len(rel)):.3f} over {len(rel)} settings")
    return out


def profile(v, nb=25):
    b = (np.arange(N) * nb) // N
    m = np.array([v[b == i].mean() for i in range(nb)])
    s = m.std(ddof=1)
    return None if s == 0 else (m - m.mean()) / s


def mean_pair_corr(Z):
    n, nb = Z.shape
    S = Z.sum(0)
    return ((S ** 2).sum() - (Z ** 2).sum()) / ((nb - 1) * n * (n - 1))


def test_b(nb=25, nperm=500):
    res = {}
    zs = []
    for job in META:
        Q = qarrays(job)
        mats = []
        for q, byk in Q.items():
            V = np.array([byk[k] for k in sorted(byk)])
            if len(V) < 10:
                continue
            P = V.reshape(len(V), nb, N // nb).mean(2)
            sd = P.std(1, ddof=1)
            P = P[sd > 0]
            if len(P) < 10:
                continue
            mats.append((P - P.mean(1, keepdims=True)) / P.std(1, ddof=1, keepdims=True))
        stat = float(np.mean([mean_pair_corr(Z) for Z in mats]))
        nulls = []
        for _ in range(nperm):
            vals = []
            for Z in mats:
                sh = RNG.integers(0, nb, len(Z))
                idx = (np.arange(nb)[None, :] + sh[:, None]) % nb
                vals.append(mean_pair_corr(np.take_along_axis(Z, idx, 1)))
            nulls.append(np.mean(vals))
        nulls = np.array(nulls)
        z = (stat - nulls.mean()) / nulls.std(ddof=1)
        zs.append(z)
        res[job] = {"stat": stat, "null_mean": float(nulls.mean()), "null_sd": float(nulls.std(ddof=1)), "z": float(z),
                    "p": float((np.sum(nulls >= stat) + 1) / (nperm + 1)), "qubits": len(mats)}
        print(f"B: {job}: per-qubit same-job piece corr {stat:+.4f} null {nulls.mean():+.4f}+-{nulls.std(ddof=1):.4f} z {z:+.1f}", flush=True)
    res["stouffer_z"] = float(np.sum(zs) / np.sqrt(len(zs)))
    print("B Stouffer z", res["stouffer_z"])
    return res


def test_c():
    """Under H2 (pieces sequential), per-qubit drift appears as a trend of piece means in k."""
    res = {}
    allz = []
    for job in META:
        Q = qarrays(job)
        zq = []
        for q, byk in Q.items():
            ks = sorted(byk)
            if len(ks) < 10:
                continue
            m = np.array([byk[k].mean() for k in ks])
            se = np.array([byk[k].std(ddof=1) / np.sqrt(N) for k in ks])
            if np.any(se == 0):
                continue
            setting = np.array([META[job]["pieces"][str(k)]["i"] for k in ks])
            t = (np.array(ks) + 0.5) / len(META[job]["pieces"])
            # per-setting intercepts, common slope in t
            num = den = 0.0
            for s in np.unique(setting):
                sel = setting == s
                if sel.sum() < 2:
                    continue
                w = 1 / se[sel] ** 2
                tb = (w * t[sel]).sum() / w.sum()
                yb = (w * m[sel]).sum() / w.sum()
                num += (w * (t[sel] - tb) * (m[sel] - yb)).sum()
                den += (w * (t[sel] - tb) ** 2).sum()
            if den == 0:
                continue
            zq.append(num / den * np.sqrt(den))
        zq = np.array(zq)
        res[job] = {"qubits": len(zq), "mean_z2": float((zq ** 2).mean()), "n_absz_gt3": int((np.abs(zq) > 3).sum())}
        allz += list(zq)
    allz = np.array(allz)
    res["all"] = {"n": len(allz), "mean_z2": float((allz ** 2).mean()), "n_absz_gt3": int((np.abs(allz) > 3).sum()),
                  "expected_gt3": float(len(allz) * 2 * stats.norm.sf(3))}
    print("C: piece-position trend per qubit (H2 signature):", res["all"])
    return res


def test_d():
    """Same per-qubit trend but in shot index j (H1 signature), unbinned, for comparison with C."""
    allz = []
    for job in META:
        Q = qarrays(job)
        for q, byk in Q.items():
            V = np.array([byk[k] for k in sorted(byk)])
            if len(V) < 10:
                continue
            Vc = V - V.mean(1, keepdims=True)
            t = (np.arange(N) + 0.5) / N - 0.5
            var = (Vc - 0).var()
            if var == 0:
                continue
            slope = (Vc * t).sum() / (len(V) * (t ** 2).sum())
            se = np.sqrt(var / (len(V) * (t ** 2).sum()))
            allz.append(slope / se)
    allz = np.array(allz)
    r = {"n": len(allz), "mean_z2": float((allz ** 2).mean()), "n_absz_gt3": int((np.abs(allz) > 3).sum()),
         "expected_gt3": float(len(allz) * 2 * stats.norm.sf(3))}
    print("D: shot-index trend per qubit (H1 signature):", r)
    return r


if __name__ == "__main__":
    out = {"A": test_a(), "C": test_c(), "D": test_d(), "B": test_b()}
    (HERE / "v_order.json").write_text(json.dumps(out, indent=1, default=float))
