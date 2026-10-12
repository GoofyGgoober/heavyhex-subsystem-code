"""Uncertainty and robustness of the hidden Markov fits behind the leaked state's persistence.

Data and model are hmm.py's: every flag and relay
series of Z memory at 4, 6 and 8 rounds of each job (320 series, 38 physical qubits, 15
jobs), a two-state hidden Markov model per series

    P(persistent at round 0) = pi, onset a = P(N -> P) per round, exit b = P(P -> N) per round,
    P(fires | N) = e_N, P(fires | P) = e_P (the paper's e_L),

fitted by Baum-Welch from hmm.em's start (pi .02, a .01, b .3, e_N = mean rate,
e_P .7) with its 300-iteration cap and 1e-7 tolerance. The paper's inputs are medians
over the 320 series: lifetime = 7.9 us x median(1/b) = 35.4 us, leaked readout =
median(e_P) = 0.51.

The EM here (em_batch) is hmm.em, run on the counts of each distinct 0/1 pattern (at most
2^8 per length) and for all series at once; it is the same algorithm and reproduces
results/hmm.json (checked, step 1). It makes the following affordable:

  1. reproduction of every series' saved fit;
  2. convergence and multiple starts: how many series stop at the 300-iteration cap, and
     the paper's start plus 6 random starts run to convergence (3000 iterations, tol 1e-9);
  3. shot bootstrap (B replicates): resample the shots of each circuit (the same shots for
     every series of a job, so their dependence is kept), refit every series (warm start at
     the full-data fit run to convergence; up to 1000 iterations, tol 1e-9), recompute the
     medians;
  4. cluster bootstraps of the medians over the 38 physical qubits and over the 15 jobs
     (resampling whole clusters of series), alone and combined with step 3 (two-stage:
     clusters resampled within each shot-bootstrap replicate), the interval the paper quotes;
  5. sensitivity to the sequence lengths fitted (8-round only; 4 and 6 only);
  6. posterior predictive checks on the 8-round series: re-firing excess at lags 1-6 and
     the run-length distribution, device against shots simulated from each fitted HMM;
  7. an alternative: a shot-level two-component mixture (the state never changes within a
     shot, a = b = 0) and a one-state model, by log-likelihood.

Heavy: hours (the random starts and the bootstrap; both are checkpointed in cache/hmm/, so
a rerun resumes).

    python docs/analysis/hmm_bootstrap.py [--boot 100] [--workers 2]

Writes results/hmm_bootstrap.json. The 0/1 patterns of the device series are cached in
cache/hmm_series.npz.
"""

from __future__ import annotations

import os

# one BLAS thread per process (OpenBLAS otherwise commits ~0.5 GB per process on Windows)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from common import CACHE, RESULTS, ROUND_US, WORKERS

LENGTHS = (4, 6, 8)
SERIES = CACHE / "hmm_series.npz"
PARAMS = ("pi", "a", "b", "e_N", "e_L")


# ------------------------------------------------------------------ data
def load_series():
    """codes[T]: S x shots pattern codes (bit t = fired at round t+1); meta per series."""
    if SERIES.exists():
        z = np.load(SERIES, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        return {T: z[f"codes{T}"] for T in LENGTHS}, meta
    from common import jobs, memory_index, readout_series

    codes = {T: [] for T in LENGTHS}
    meta = []
    for job in jobs():
        per = defaultdict(dict)
        for T in LENGTHS:
            idx = memory_index("Z", T)
            model, records, labels = job.model(idx)
            chip, _ = job.chip_detectors(idx, model, records)
            for key, cols in readout_series(labels).items():
                x = chip[:, cols]
                assert x.shape[1] == T
                per[key][T] = (x.astype(np.int64) << np.arange(T)).sum(1).astype(np.int16)
        for key, d in per.items():
            meta.append(
                {
                    "job": job.name,
                    "day": job.day,
                    "rep": job.rep,
                    "patch": key[0],
                    "readout": key[1],
                    "k": key[2],
                    "qubit": job.readout_qubit(*key),
                }
            )
            for T in LENGTHS:
                codes[T].append(d[T])
        print(job.name, "loaded", flush=True)
    codes = {T: np.stack(v) for T, v in codes.items()}
    SERIES.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(SERIES, meta=json.dumps(meta), **{f"codes{T}": codes[T] for T in LENGTHS})
    return codes, meta


def counts_of(codes: np.ndarray, T: int, idx: np.ndarray | None = None) -> np.ndarray:
    """S x 2^T pattern counts (optionally of the shots idx)."""
    c = codes if idx is None else codes[:, idx]
    P = 1 << T
    off = (np.arange(len(c)) * P)[:, None]
    return (
        np.bincount((c.astype(np.int64) + off).ravel(), minlength=len(c) * P)
        .reshape(len(c), P)
        .astype(np.float64)
    )


def patterns(T: int) -> np.ndarray:
    return ((np.arange(1 << T)[:, None] >> np.arange(T)) & 1).astype(bool)  # P x T


# ------------------------------------------------------------------ EM
def em_batch(counts: dict, init: dict, iters=300, tol=1e-7, fix_ab: bool = False):
    """hmm.em for many series at once, on pattern counts.

    counts {T: S x 2^T}, init {param: S array}. Same updates, clipping and stopping rule as
    hmm.em (a series stops after the update in which its log-likelihood rose by
    less than tol); the two-state forward-backward is written out state by state. fix_ab
    keeps a = b = 0 (a shot-level mixture). Returns params, the log-likelihood at the last
    E-step, the number of iterations of each series and which series hit the cap.
    """
    p = {k: np.array(v, dtype=np.float64).copy() for k, v in init.items()}
    S = len(p["b"])
    active = np.ones(S, bool)
    last = np.full(S, -np.inf)
    ll_out = np.zeros(S)
    n_it = np.zeros(S, int)
    # only the patterns that occur in some series
    data = {}
    for T, cnt in counts.items():
        used = np.flatnonzero(cnt.sum(0) > 0)
        data[T] = (cnt[:, used], patterns(T)[used])
    for it in range(iters):
        ia = np.flatnonzero(active)
        if not len(ia):
            break
        n = defaultdict(lambda: np.zeros(len(ia)))  # noqa: B023 (this iteration's ia)
        ll = np.zeros(len(ia))
        pi, a, b, e0, e1 = (p[k][ia][:, None] for k in PARAMS)
        for T, (cnt, pat) in data.items():
            w = cnt[ia]  # s x P
            O = pat.T  # T x P
            BN = np.where(O[:, None, :], e0[None], 1 - e0[None])  # T x s x P
            BP = np.where(O[:, None, :], e1[None], 1 - e1[None])
            aN = np.empty_like(BN)
            aP = np.empty_like(BN)
            c = np.empty_like(BN)
            xN, xP = (1 - pi) * BN[0], pi * BP[0]
            c[0] = xN + xP
            aN[0] = xN / c[0]
            aP[0] = xP / c[0]
            for t in range(1, T):
                xN = (aN[t - 1] * (1 - a) + aP[t - 1] * b) * BN[t]
                xP = (aN[t - 1] * a + aP[t - 1] * (1 - b)) * BP[t]
                c[t] = xN + xP
                aN[t] = xN / c[t]
                aP[t] = xP / c[t]
            bN = np.ones_like(BN)
            bP = np.ones_like(BN)
            for t in range(T - 2, -1, -1):
                yN, yP = BN[t + 1] * bN[t + 1], BP[t + 1] * bP[t + 1]
                bN[t] = ((1 - a) * yN + a * yP) / c[t + 1]
                bP[t] = (b * yN + (1 - b) * yP) / c[t + 1]
            gN, gP = aN * bN, aP * bP
            z = gN + gP
            gN, gP = gN / z, gP / z
            xi01 = aN[:-1] * a * BP[1:] * bP[1:] / c[1:]
            xi10 = aP[:-1] * b * BN[1:] * bN[1:] / c[1:]
            ll += (w * np.log(c).sum(0)).sum(1)
            Of = O[:, None, :]
            n["pi"] += (w * gP[0]).sum(1)
            n["S"] += w.sum(1)
            n["01"] += (w * xi01.sum(0)).sum(1)
            n["0"] += (w * gN[:-1].sum(0)).sum(1)
            n["10"] += (w * xi10.sum(0)).sum(1)
            n["1"] += (w * gP[:-1].sum(0)).sum(1)
            n["f0"] += (w * (gN * Of).sum(0)).sum(1)
            n["g0"] += (w * gN.sum(0)).sum(1)
            n["f1"] += (w * (gP * Of).sum(0)).sum(1)
            n["g1"] += (w * gP.sum(0)).sum(1)
        new = {
            "pi": n["pi"] / n["S"],
            "a": n["01"] / np.maximum(n["0"], 1e-12),
            "b": n["10"] / np.maximum(n["1"], 1e-12),
            "e_N": n["f0"] / np.maximum(n["g0"], 1e-12),
            "e_L": n["f1"] / np.maximum(n["g1"], 1e-12),
        }
        for k in PARAMS:
            p[k][ia] = np.clip(new[k], 1e-6, 1 - 1e-6)
        if fix_ab:
            p["a"][ia] = 0.0
            p["b"][ia] = 0.0
        ll_out[ia] = ll
        n_it[ia] = it + 1
        done = ll - last[ia] < tol
        last[ia] = ll
        active[ia[done]] = False
    return p, ll_out, n_it, active  # active: stopped by the iteration cap


def paper_init(counts: dict) -> dict:
    S = len(next(iter(counts.values())))
    rate = np.mean(
        [(cnt * patterns(T).sum(1)[None]).sum(1) / (cnt.sum(1) * T) for T, cnt in counts.items()],
        axis=0,
    )
    return {
        "pi": np.full(S, 0.02),
        "a": np.full(S, 0.01),
        "b": np.full(S, 0.3),
        "e_N": np.maximum(rate, 1e-3),
        "e_L": np.full(S, 0.7),
    }


def loglik(counts: dict, p: dict) -> np.ndarray:
    """Log-likelihood of each series at parameters p (one E-step, no update)."""
    _, ll, _, _ = em_batch(counts, p, iters=1)
    return ll


def relabel(p: dict) -> dict:
    """States labelled so that e_P > e_N."""
    sw = p["e_L"] < p["e_N"]
    q = {k: v.copy() for k, v in p.items()}
    q["pi"][sw] = 1 - p["pi"][sw]
    q["a"][sw], q["b"][sw] = p["b"][sw], p["a"][sw]
    q["e_N"][sw], q["e_L"][sw] = p["e_L"][sw], p["e_N"][sw]
    return q


def summary(p: dict) -> dict:
    b = p["b"]
    return {
        "b": float(np.median(b)),
        "life_rounds": float(np.median(1 / np.maximum(b, 1e-6))),
        "life_us": float(np.median(1 / np.maximum(b, 1e-6)) * ROUND_US),
        "life_us_cont": float(np.median(-ROUND_US / np.log(1 - b))),
        "e_P": float(np.median(p["e_L"])),
        "a": float(np.median(p["a"])),
        "pi": float(np.median(p["pi"])),
        "e_N": float(np.median(p["e_N"])),
    }


KEYS = ("b", "life_rounds", "life_us", "life_us_cont", "e_P", "a", "pi", "e_N")


def ci(vals: list[dict], q=(2.5, 97.5)) -> dict:
    return {
        k: [
            float(np.percentile([v[k] for v in vals], q[0])),
            float(np.percentile([v[k] for v in vals], q[1])),
        ]
        for k in KEYS
    }


def cluster_medians(p: dict, groups: list[np.ndarray], rng, B: int) -> list[dict]:
    out = []
    for _ in range(B):
        ii = np.concatenate([groups[g] for g in rng.integers(0, len(groups), len(groups))])
        out.append(summary({k: v[ii] for k, v in p.items()}))
    return out


# ------------------------------------------------------------------ checks
def refire(x: np.ndarray, lag: int) -> float:
    now, nxt = x[:, :-lag], x[:, lag:]
    return (now & nxt).sum() / max(now.sum(), 1) - (~now & nxt).sum() / max((~now).sum(), 1)


def run_hist(x: np.ndarray, T: int) -> np.ndarray:
    """Share of shots whose longest run of consecutive firings is 0..T."""
    run = np.zeros(len(x), int)
    best = np.zeros(len(x), int)
    for t in range(x.shape[1]):
        run = np.where(x[:, t], run + 1, 0)
        best = np.maximum(best, run)
    return np.bincount(best, minlength=T + 1) / len(x)


def sample_hmm(p, i, shots, T, rng):
    s = rng.random(shots) < p["pi"][i]
    out = np.empty((shots, T), bool)
    for t in range(T):
        if t:
            u = rng.random(shots)
            s = np.where(s, u >= p["b"][i], u < p["a"][i])
        out[:, t] = rng.random(shots) < np.where(s, p["e_L"][i], p["e_N"][i])
    return out


def decode(codes: np.ndarray, T: int) -> np.ndarray:
    return ((codes[..., None].astype(np.int64) >> np.arange(T)) & 1).astype(bool)


# ------------------------------------------------------------------ parallel parts
CONV_ITERS, CONV_TOL, BOOT_ITERS = 3000, 1e-9, 1000
CKPT = CACHE / "hmm"  # per-start and per-replicate fits, so that a rerun resumes
_W: dict = {}


def _worker_init():
    codes, meta = load_series()
    _W["codes"], _W["meta"] = codes, meta
    _W["counts"] = {T: counts_of(codes[T], T) for T in LENGTHS}


def _start(i):
    """i = -1: the paper's start; else random start i. Run to convergence (checkpointed in cache/)."""
    path = CKPT / f"start{i}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in PARAMS}, z["ll"], z["cap"]
    counts = _W["counts"]
    S = len(_W["meta"])
    if i < 0:
        init = paper_init(counts)
    else:
        rng = np.random.default_rng(100 + i)
        init = {
            "pi": rng.uniform(0.005, 0.3, S),
            "a": rng.uniform(0.003, 0.15, S),
            "b": rng.uniform(0.05, 0.8, S),
            "e_N": rng.uniform(0.01, 0.15, S),
            "e_L": rng.uniform(0.2, 0.95, S),
        }
    ps, ll, _, cap = em_batch(counts, init, iters=CONV_ITERS, tol=CONV_TOL)
    ps = relabel(ps)
    _save(path, **ps, ll=ll, cap=cap)
    return ps, ll, cap


def _save(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    with open(part, "wb") as fh:
        np.savez(fh, **arrays)
    os.replace(part, path)


def _boot(r):
    """Replicate r: resample each circuit's shots (the same shots for every series of a job) and refit every series."""
    path = CKPT / f"boot{r}.npz"
    if path.exists():
        z = np.load(path)
        return {k: z[k] for k in PARAMS}
    codes, meta = _W["codes"], _W["meta"]
    rng = np.random.default_rng([3, r])
    N = codes[8].shape[1]
    cnt = {T: np.empty((len(meta), 1 << T)) for T in LENGTHS}
    for j in sorted({m["job"] for m in meta}):
        ii = np.array([i for i, m in enumerate(meta) if m["job"] == j])
        for T in LENGTHS:
            cnt[T][ii] = counts_of(codes[T][ii], T, rng.integers(0, N, N))
    # warm start at the full-data fit run to convergence (the paper's start, start-1.npz)
    z = np.load(CKPT / "start-1.npz")
    pb, _, _, _ = em_batch(cnt, {k: z[k] for k in PARAMS}, iters=BOOT_ITERS, tol=CONV_TOL)
    pb = relabel(pb)
    _save(path, **pb)
    return pb


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--boot", type=int, default=100)
    ap.add_argument("--starts", type=int, default=6)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    codes, meta = load_series()
    S = len(meta)
    counts = {T: counts_of(codes[T], T) for T in LENGTHS}
    print(
        f"{S} series, {len({m['qubit'] for m in meta})} physical qubits, {len({m['job'] for m in meta})} jobs; "
        f"shots per length {codes[8].shape[1]}"
    )

    # 1. reproduction
    p0, ll0, _it0, capped = em_batch(counts, paper_init(counts))
    saved = json.loads((RESULTS / "hmm.json").read_text())
    key = lambda m: (m["job"], m["patch"], m["readout"], m["qubit"])  # noqa: E731
    sv = {key(r): r["chip"] for r in saved}
    dmax = max(abs(p0[k][i] - sv[key(m)][k]) for i, m in enumerate(meta) for k in PARAMS)
    full = summary(p0)
    print(f"\n1. reproduction of results/hmm.json: max |difference| over all parameters {dmax:.1e}")
    print("   medians: " + "  ".join(f"{k} {full[k]:.4g}" for k in KEYS))

    # 2. convergence and starts (in parallel: the paper's start and random starts, run to convergence)
    print(
        f"\n2. {capped.sum()} of {S} series stopped at the 300-iteration cap without meeting the tolerance"
    )
    with mp.Pool(a.workers, initializer=_worker_init) as pool:
        runs = pool.map(
            _start, range(-1, a.starts), chunksize=1
        )  # the slow part: hours on a loaded machine
        print("   starts done", flush=True)
        boot_p = []
        for r, pb in enumerate(pool.imap(_boot, range(a.boot), chunksize=1), 1):
            boot_p.append(pb)
            if r % 50 == 0:
                print(f"   bootstrap {r}/{a.boot}", flush=True)
    pc, llc, capc = runs[0]
    best_p, best_ll = {k: v.copy() for k, v in pc.items()}, llc.copy()
    lls = [llc]
    for ps, lls_, _ in runs[1:]:
        lls.append(lls_)
        better = lls_ > best_ll
        for k in PARAMS:
            best_p[k][better] = ps[k][better]
        best_ll[better] = lls_[better]
    gain = best_ll - ll0
    conv = summary(pc)
    bst = summary(best_p)
    lls = np.array(lls)
    n_modes = (lls > best_ll[None] - 0.5).sum(0)
    print(
        f"   paper start run to convergence (tol 1e-9, cap {CONV_ITERS}): still capped in {capc.sum()}; medians b {conv['b']:.4f}, "
        f"lifetime {conv['life_us']:.1f} us, e_P {conv['e_P']:.3f}, a {conv['a']:.4f}"
    )
    print(
        f"   best of {a.starts + 1} starts: log-likelihood above the published fit by > 0.5 in {np.mean(gain > 0.5):.1%} "
        f"of series (> 2 in {np.mean(gain > 2):.1%}; max {gain.max():.1f}); medians b {bst['b']:.4f}, lifetime {bst['life_us']:.1f} us, "
        f"e_P {bst['e_P']:.3f}, a {bst['a']:.4f}"
    )
    print(
        f"   per series |b_best - b_published|: median {np.median(np.abs(best_p['b'] - p0['b'])):.4f}, "
        f"90th pct {np.percentile(np.abs(best_p['b'] - p0['b']), 90):.4f}; starts reaching within 0.5 of the best: median {np.median(n_modes):.0f} of {a.starts + 1}"
    )

    # 3. shot bootstrap (run above, in the pool)
    shot_meds = [summary(pb) for pb in boot_p]
    se_b = np.std([pb["b"] for pb in boot_p], axis=0, ddof=1)
    se_e = np.std([pb["e_L"] for pb in boot_p], axis=0, ddof=1)
    se_a = np.std([pb["a"] for pb in boot_p], axis=0, ddof=1)
    ci_shot = ci(shot_meds)
    print(
        f"\n3. shot bootstrap ({a.boot} replicates, every series refitted): 95% intervals of the medians "
        f"(published value, then the bootstrap median)"
    )
    for k in KEYS:
        print(
            f"   {k:12s} {full[k]:.4g}  {np.median([m[k] for m in shot_meds]):.4g}  [{ci_shot[k][0]:.4g}, {ci_shot[k][1]:.4g}]"
        )
    print(
        f"   per series: SE(b)/b median {np.median(se_b / p0['b']):.0%} (IQR {np.percentile(se_b / p0['b'], 25):.0%}-"
        f"{np.percentile(se_b / p0['b'], 75):.0%}); SE(e_P) median {np.median(se_e):.3f}; SE(a)/a median {np.median(se_a / p0['a']):.0%}"
    )

    # 4. cluster bootstraps
    out4 = {}
    for cl in ("qubit", "job"):
        g = defaultdict(list)
        for i, m in enumerate(meta):
            g[m[cl]].append(i)
        groups = [np.array(v) for v in g.values()]
        rngc = np.random.default_rng(11)
        alone = cluster_medians(p0, groups, rngc, 4000)
        two = []
        for pb in boot_p:
            two += cluster_medians(pb, groups, rngc, max(1, 4000 // a.boot))
        out4[cl] = {"G": len(groups), "alone": ci(alone), "two_stage": ci(two)}
        print(
            f"\n4. cluster bootstrap over the {len(groups)} {cl}s: 95% intervals of the medians (alone | with the shot bootstrap)"
        )
        for k in KEYS:
            print(
                f"   {k:12s} {full[k]:.4g}  [{out4[cl]['alone'][k][0]:.4g}, {out4[cl]['alone'][k][1]:.4g}] | "
                f"[{out4[cl]['two_stage'][k][0]:.4g}, {out4[cl]['two_stage'][k][1]:.4g}]"
            )
    life = 1 / p0["b"] * ROUND_US
    print(
        f"   spread across series (heterogeneity, not uncertainty): 7.9 us / b IQR {np.percentile(life, 25):.0f}-{np.percentile(life, 75):.0f} us, "
        f"10-90% {np.percentile(life, 10):.0f}-{np.percentile(life, 90):.0f} us; e_P IQR {np.percentile(p0['e_L'], 25):.2f}-{np.percentile(p0['e_L'], 75):.2f}"
    )
    per_q = defaultdict(list)
    for i, m in enumerate(meta):
        per_q[m["qubit"]].append(p0["b"][i])
    qmed = np.array([np.median(v) for v in per_q.values()])
    print(
        f"   per physical qubit median b: median {np.median(qmed):.3f}, IQR {np.percentile(qmed, 25):.3f}-{np.percentile(qmed, 75):.3f} "
        f"(7.9 us / b: {ROUND_US / np.percentile(qmed, 75):.0f}-{ROUND_US / np.percentile(qmed, 25):.0f} us); series per qubit "
        f"median {np.median([len(v) for v in per_q.values()]):.0f}, max {max(len(v) for v in per_q.values())}"
    )

    # 5. lengths
    out5 = {}
    for name, Ts in (("8 only", (8,)), ("4 and 6 only", (4, 6))):
        cs = {T: counts[T] for T in Ts}
        ps, _, _, _ = em_batch(cs, paper_init(cs))
        out5[name] = summary(ps)
        print(
            f"\n5. lengths {name}: medians b {out5[name]['b']:.4f} (lifetime {out5[name]['life_us']:.1f} us), "
            f"e_P {out5[name]['e_P']:.3f}, a {out5[name]['a']:.4f}"
        )

    # 6. posterior predictive on the 8-round series
    rng = np.random.default_rng(6)
    lags = (1, 2, 3, 4, 5, 6)
    dev_r, hmm_r, dev_h, hmm_h = [], [], [], []
    for i in range(S):
        x = decode(codes[8][i], 8)
        sim = sample_hmm(p0, i, 20000, 8, rng)
        dev_r.append([refire(x, k) for k in lags])
        hmm_r.append([refire(sim, k) for k in lags])
        dev_h.append(run_hist(x, 8))
        hmm_h.append(run_hist(sim, 8))
    dev_r, hmm_r = np.mean(dev_r, 0), np.mean(hmm_r, 0)
    dev_h, hmm_h = np.mean(dev_h, 0), np.mean(hmm_h, 0)
    print("\n6. posterior predictive, 8-round series (mean over series):")
    print("   re-firing excess at lag " + " ".join(str(k) for k in lags) + " (percentage points)")
    print("     device " + " ".join(f"{100 * v:6.2f}" for v in dev_r))
    print("     HMM    " + " ".join(f"{100 * v:6.2f}" for v in hmm_r))
    print("   longest run of consecutive firings = 0..8 (% of shots)")
    print("     device " + " ".join(f"{100 * v:6.2f}" for v in dev_h))
    print("     HMM    " + " ".join(f"{100 * v:6.2f}" for v in hmm_h))

    # 7. alternatives
    _pm, llm, _, _ = em_batch(
        counts,
        {**paper_init(counts), "a": np.zeros(S), "b": np.zeros(S), "pi": np.full(S, 0.1)},
        iters=5000,
        tol=1e-10,
        fix_ab=True,
    )
    ll1 = np.zeros(S)
    for T in LENGTHS:
        pat = patterns(T)
        k1 = (counts[T] * pat.sum(1)[None]).sum(1)  # per series: firings at each round pooled below
        for t in range(T):
            f = (counts[T] * pat[:, t][None]).sum(1)
            n_ = counts[T].sum(1)
            q = np.clip(f / n_, 1e-9, 1 - 1e-9)
            ll1 += f * np.log(q) + (n_ - f) * np.log(1 - q)
        del k1
    print("\n7. alternatives, log-likelihood of the HMM (5 parameters) minus:")
    print(
        f"   shot-level mixture (state fixed within a shot; 3 parameters): median {np.median(best_ll - llm):.1f}, "
        f"> 10 in {np.mean(best_ll - llm > 10):.0%} of series, total {np.sum(best_ll - llm):.0f}"
    )
    print(
        f"   independent rounds, own rate per round (hmm.py's llr): median {np.median(best_ll - ll1):.1f}"
    )

    out = {
        "n_series": S,
        "boot": a.boot,
        "starts": a.starts,
        "reproduction_max_abs_diff": float(dmax),
        "published": full,
        "capped_at_300": int(capped.sum()),
        "converged_paper_start": conv,
        "best_of_starts": bst,
        "share_ll_gain_gt_0.5": float(np.mean(gain > 0.5)),
        "share_ll_gain_gt_2": float(np.mean(gain > 2)),
        "shot_bootstrap_ci95": ci_shot,
        "cluster": out4,
        "lengths": out5,
        "per_series_rel_se_b_median": float(np.median(se_b / p0["b"])),
        "per_series_se_eP_median": float(np.median(se_e)),
        "ppc": {
            "lags": list(lags),
            "device_refire": dev_r.tolist(),
            "hmm_refire": hmm_r.tolist(),
            "device_longest_run": dev_h.tolist(),
            "hmm_longest_run": hmm_h.tolist(),
        },
        "alt_llr_vs_mixture_median": float(np.median(best_ll - llm)),
        "alt_llr_vs_independent_median": float(np.median(best_ll - ll1)),
        "series": [
            {
                **m,
                **{k: float(p0[k][i]) for k in PARAMS},
                "se_b": float(se_b[i]),
                "se_e_P": float(se_e[i]),
            }
            for i, m in enumerate(meta)
        ],
    }
    (RESULTS / "hmm_bootstrap.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\nwrote", RESULTS / "hmm_bootstrap.json")


if __name__ == "__main__":
    main()
