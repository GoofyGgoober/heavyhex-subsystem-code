"""What the leakage scripts share: the fit's jobs and statistics, its inputs, and a cache of
exact simulations.

The leakage model (Sec. leakage): every qubit leaks with probability `leak` per CZ; the
partner of a leaked qubit's CZ gets an X error with probability `strength` and leaks with
probability `spread`; a leaked qubit returns after an exponential lifetime and reads 1 with
probability `read` (the hidden Markov fits, hmm.py); each reset and readout has IBM's
reported error for the job's day where recorded (ibm_errors). It is fitted to six detection
statistics of the 8-round circuits of four jobs (FIT_JOBS), and simulated either by
frame_sim.py or, exactly, by exact_sim.py.

  stats(det, labels)       the six statistics as fitted (pooled co-streak excess)
  stats_full(det, labels)  the same plus the round-conditioned versions (per slot) and more
  Point, ensure, load      exact simulations of one circuit at one parameter point, run in a
                           worker pool and kept in cache/exact_* (rebuilt on demand)
"""

from __future__ import annotations

import os

# one BLAS thread per process (OpenBLAS otherwise commits about 0.5 GB per process)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import collections
import itertools
import json
import multiprocessing as mp
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from common import (
    CACHE,
    RESULTS,
    ROUND_US,
    WORKERS,
    data_read,
    jobs,
    memory_index,
    readout_series,
    z4_of,
)
from properties import init_error, readout_split

FIT_JOBS = ("2026-10-08-r1", "2026-10-08-r2-d5", "2026-10-08-r2-d3", "2026-10-06-r1")
CASES = [(j, b) for j in FIT_JOBS for b in "XZ"]  # the fit's eight circuits, 8 rounds each
GRID_SHOTS, GRID_SEED = 8000, 21
PAPER_GRID = list(itertools.product((0.001, 0.002, 0.003), (0.25, 0.5, 1.0), (0.0, 0.05)))
BEST_PUBLISHED = (0.002, 0.25, 0.05)
# The published grid's points near its optimum (strength 25%, and 50% at leak 0.2%); the
# strength-100% points lost by a factor of 10 or more in every run.
EXACT_SUBSET = [
    (0.001, 0.25, 0.0),
    (0.001, 0.25, 0.05),
    (0.002, 0.25, 0.0),
    (0.002, 0.25, 0.05),
    (0.003, 0.25, 0.0),
    (0.003, 0.25, 0.05),
    (0.002, 0.5, 0.0),
    (0.002, 0.5, 0.05),
]
SIM = os.environ.get("LEAKFIT_SIM", "x")  # "x": exact_sim.py; "t": tableau_ref.py (slow)


# ------------------------------------------------------------------ inputs
def hmm_inputs(jobs_subset=None) -> tuple[float, float]:
    """(lifetime in us, leaked readout): medians of the hidden Markov fits, optionally of some jobs."""
    hmm = json.loads((RESULTS / "hmm.json").read_text())
    if jobs_subset is not None:
        hmm = [r for r in hmm if r["job"] in jobs_subset]
    life = float(np.median([1 / max(r["chip"]["b"], 1e-6) for r in hmm])) * ROUND_US
    read = float(np.median([r["chip"]["e_L"] for r in hmm]))
    return life, read


def ibm_errors(day: str) -> tuple[dict, dict]:
    """IBM's reset error (init_error) and (P(1|0), P(0|1)) per qubit, for the day's calibration
    (properties.py says which are recorded). A qubit missing from either keeps the model's error,
    the readout error in force."""
    return init_error(day), readout_split(day)


class AllQubits(dict):
    """Every qubit has a leak rate: its own if measured, a default otherwise."""

    def __init__(self, leak):
        super().__init__()
        self._leak = leak

    def get(self, q, default=None):
        return self._leak[q]

    def __bool__(self):
        return True


# ------------------------------------------------------------------ the six statistics
FIT_KEYS = ["X stab", "Z stab", "Z flag", "Z repeat", "Z co_data", "Z co_gauge"]
VARIANTS = {
    "pooled": {},
    "slot": {"Z co_data": "Z co_data_slot", "Z co_gauge": "Z co_gauge_slot"},
    "slot+repeat": {
        "Z co_data": "Z co_data_slot",
        "Z co_gauge": "Z co_gauge_slot",
        "Z repeat": "Z repeat_slot",
    },
}


def stats(det: np.ndarray, labels) -> dict:
    """Mean detection rate of the stabilizers and of the flags and relays, the flags' and
    relays' repeat excess P(fire at r+1 | fired at r) - P(fire at r+1 | quiet at r), and the
    co-streak excess of pairs that read a common data qubit and of pairs on one Z gauge."""
    out = {}
    stab = [c for c, (p, r, n) in enumerate(labels) if " " not in n and 1 <= r <= 7]
    out["stab"] = float(det[:, stab].mean())
    series = readout_series(labels)
    if series:
        flags = [c for cols in series.values() for c in cols]
        out["flag"] = float(det[:, flags].mean())
        rep = []
        st = {}
        for key, cols in series.items():
            x = det[:, cols]
            now, nxt = x[:, :-1], x[:, 1:]
            rep.append((now & nxt).sum() / max(now.sum(), 1) - (~now & nxt).sum() / (~now).sum())
            st[key] = now & nxt
        out["repeat"] = float(np.mean(rep))
        share, gauge = [], []
        for a, b in itertools.combinations(series, 2):
            if a[0] != b[0]:
                continue
            v = float((st[a] & st[b]).mean() - st[a].mean() * st[b].mean())
            if data_read(*a) & data_read(*b):
                share.append(v)
            elif z4_of(a[0], a[1]) and z4_of(a[0], a[1]) == z4_of(b[0], b[1]):
                gauge.append(v)
        out["co_data"] = float(np.mean(share)) if share else np.nan
        out["co_gauge"] = float(np.mean(gauge)) if gauge else np.nan
    return out


def stats_full(det: np.ndarray, labels, check: bool = True) -> dict:
    """stats() plus

    co_data_slot, co_gauge_slot  the co-streak excess P(A&B | slot) - P(A|slot) P(B|slot) in
                                 each of the 7 streak slots (rounds 2..8), averaged over the
                                 slots, per pair, then over the pairs of the class (costreak.py)
    repeat_slot                  the repeat excess per round r, averaged over r
    co_unrel(_slot)              the same for unrelated same-patch pairs (not fitted)
    profile                      the mean streak probability in each slot
    """
    out = {}
    stab = [c for c, (p, r, n) in enumerate(labels) if " " not in n and 1 <= r <= 7]
    out["stab"] = float(det[:, stab].mean())
    series = readout_series(labels)
    if series:
        flags = [c for cols in series.values() for c in cols]
        out["flag"] = float(det[:, flags].mean())
        rep, rep_slot, st = [], [], {}
        for key, cols in series.items():
            x = det[:, cols]
            now, nxt = x[:, :-1], x[:, 1:]
            rep.append((now & nxt).sum() / max(now.sum(), 1) - (~now & nxt).sum() / (~now).sum())
            per_r = (now & nxt).sum(0) / np.maximum(now.sum(0), 1) - (~now & nxt).sum(
                0
            ) / np.maximum((~now).sum(0), 1)
            rep_slot.append(per_r.mean())
            st[key] = now & nxt
        out["repeat"] = float(np.mean(rep))
        out["repeat_slot"] = float(np.mean(rep_slot))
        acc = collections.defaultdict(list)
        for a, b in itertools.combinations(series, 2):
            if a[0] != b[0]:
                continue
            sa, sb = st[a], st[b]
            pooled = float((sa & sb).mean() - sa.mean() * sb.mean())
            slot = float(np.mean((sa & sb).mean(0) - sa.mean(0) * sb.mean(0)))
            if data_read(*a) & data_read(*b):
                cls = "co_data"
            elif z4_of(a[0], a[1]) and z4_of(a[0], a[1]) == z4_of(b[0], b[1]):
                cls = "co_gauge"
            else:
                cls = "co_unrel"
            acc[cls].append(pooled)
            acc[cls + "_slot"].append(slot)
        for cls in ("co_data", "co_gauge", "co_unrel"):
            for suffix in ("", "_slot"):
                v = acc.get(cls + suffix)
                out[cls + suffix] = float(np.mean(v)) if v else np.nan
        out["profile"] = np.mean([s.mean(0) for s in st.values()], axis=0).tolist()
    if check:
        ref = stats(det, labels)
        for k, v in ref.items():
            assert (np.isnan(v) and np.isnan(out[k])) or abs(v - out[k]) <= 1e-12 * max(
                1.0, abs(v)
            ), (k, v, out[k])
    return out


class Prepared:
    """stats_full() of one detector array under shot weights w (bootstrap counts), fast.

    Precomputes per-shot sums so that each statistic is a weighted sum over shots; with
    w = 1 it equals stats_full(det, labels), and with w = bincount(idx) it equals
    stats_full(det[idx], labels) (both checked in selftest()).
    """

    def __init__(self, det: np.ndarray, labels) -> None:
        self.N = len(det)
        stab = [c for c, (p, r, n) in enumerate(labels) if " " not in n and 1 <= r <= 7]
        self.stab = det[:, stab].sum(1).astype(np.float64)
        self.n_stab = len(stab)
        series = readout_series(labels)
        self.has_series = bool(series)
        if not series:
            return
        keys = list(series)
        x = np.stack([det[:, series[k]] for k in keys], axis=2)  # N x T x S
        self.flag = x.sum((1, 2)).astype(np.float64)
        self.n_flag = x.shape[1] * x.shape[2]
        now, nxt = x[:, :-1], x[:, 1:]
        st = now & nxt  # N x 7 x S
        self.T, self.S = st.shape[1], st.shape[2]
        self.st = st.astype(np.float32)
        self.c1 = now.reshape(self.N, -1).astype(np.float32)
        self.c01 = (~now & nxt).reshape(self.N, -1).astype(np.float32)
        cls = collections.defaultdict(list)
        for i, j in itertools.combinations(range(len(keys)), 2):
            a, b = keys[i], keys[j]
            if a[0] != b[0]:
                continue
            if data_read(*a) & data_read(*b):
                cls["co_data"].append((i, j))
            elif z4_of(a[0], a[1]) and z4_of(a[0], a[1]) == z4_of(b[0], b[1]):
                cls["co_gauge"].append((i, j))
            else:
                cls["co_unrel"].append((i, j))
        self.pairs = {
            k: (np.array([p[0] for p in v]), np.array([p[1] for p in v])) for k, v in cls.items()
        }

    def stats(self, w: np.ndarray | None = None) -> dict:
        w = np.ones(self.N) if w is None else np.asarray(w, np.float64)
        W = w.sum()
        out = {"stab": float(w @ self.stab / (W * self.n_stab))}
        if not self.has_series:
            return out
        out["flag"] = float(w @ self.flag / (W * self.n_flag))
        wf = w.astype(np.float32)  # integer counts: float32 sums stay exact below 2**24
        shape = (self.T, self.S)
        s11 = (wf @ self.st.reshape(self.N, -1)).astype(np.float64).reshape(shape)  # slot x series
        s1 = (wf @ self.c1).astype(np.float64).reshape(shape)
        s01 = (wf @ self.c01).astype(np.float64).reshape(shape)
        s0 = W - s1
        rep = s11.sum(0) / np.maximum(s1.sum(0), 1) - s01.sum(0) / s0.sum(0)
        rep_slot = (s11 / np.maximum(s1, 1) - s01 / np.maximum(s0, 1)).mean(0)
        out["repeat"] = float(rep.mean())
        out["repeat_slot"] = float(rep_slot.mean())
        # joint streak counts per slot: M[t] = st_t^T diag(w) st_t
        M = np.stack(
            [
                ((self.st[:, t, :] * wf[:, None]).T @ self.st[:, t, :]).astype(np.float64)
                for t in range(self.T)
            ]
        )
        p_slot = s11 / W  # slot x series, P(streak | slot)
        pj_slot = M / W  # slot x series x series
        p_pool = p_slot.mean(0)
        pj_pool = pj_slot.mean(0)
        for cls in ("co_data", "co_gauge", "co_unrel"):
            if cls not in self.pairs:
                out[cls] = out[cls + "_slot"] = np.nan
                continue
            i, j = self.pairs[cls]
            out[cls] = float(np.mean(pj_pool[i, j] - p_pool[i] * p_pool[j]))
            out[cls + "_slot"] = float(
                np.mean((pj_slot[:, i, j] - p_slot[:, i] * p_slot[:, j]).mean(0))
            )
        out["profile"] = p_slot.mean(1).tolist()
        return out


def selftest(det: np.ndarray, labels, seed: int = 0) -> float:
    """Largest relative difference between Prepared and stats_full(), plain and resampled."""
    pr = Prepared(det, labels)
    idx = np.random.default_rng(seed).integers(0, len(det), len(det))
    worst = 0.0
    for a, b in (
        (pr.stats(), stats_full(det, labels)),
        (pr.stats(np.bincount(idx, minlength=len(det))), stats_full(det[idx], labels)),
    ):
        for k in b:
            if k == "profile":
                d = np.max(np.abs(np.array(a[k]) - np.array(b[k])))
            elif np.isnan(b[k]):
                d = 0.0 if np.isnan(a[k]) else np.inf
            else:
                d = abs(a[k] - b[k]) / max(abs(b[k]), 1e-6)
            worst = max(worst, d)
    return worst


def merge(per_case) -> dict:
    """Mean over the cases of each '<basis> <stat>' (nanmean, as the published fit)."""
    tot = collections.defaultdict(list)
    for s in per_case:
        for k, v in s.items():
            tot[k].append(v)
    return {
        k: (np.mean(v, axis=0).tolist() if k.endswith("profile") else float(np.nanmean(v)))
        for k, v in tot.items()
    }


def loss(sim: dict, dev: dict, variant: str = "pooled") -> tuple[float, dict]:
    """Root-mean-square relative deviation of the six statistics from the device's."""
    sub = VARIANTS[variant]
    rel = {}
    for k in FIT_KEYS:
        kk = sub.get(k, k)
        rel[k] = (sim[kk] - dev[kk]) / abs(dev[kk])
    return float(np.sqrt(np.mean([v**2 for v in rel.values()]))), rel


# ------------------------------------------------------------------ the fit's circuits and grid
def device_cases() -> dict:
    """{(job, basis): (chip detectors, labels)} for the fit's eight circuits."""
    out = {}
    for j in [j for j in jobs() if j.name in FIT_JOBS]:
        for b in "XZ":
            model, records, labels = j.model(memory_index(b, 8))
            chip, _ = j.chip_detectors(memory_index(b, 8), model, records)
            out[(j.name, b)] = (chip, labels)
    return out


def case_stats(arrays: dict, labels: dict, check=False) -> dict:
    return {c: stats_full(arrays[c], labels[c], check=check) for c in CASES}


def merged(per_case: dict) -> dict:
    return merge({f"{b} {k}": v for k, v in per_case[(j, b)].items()} for j, b in CASES)


def exact_grid(seed: int = GRID_SEED, shots: int = GRID_SHOTS) -> dict:
    """{(leak, strength, spread): simulated statistics} of the exact grid (leakage_grid.py)."""
    return {
        (q["leak"], q["strength"], q["spread"]): q["sim"]
        for q in exact_grid_all()
        if q["seed"] == seed and q["shots"] == shots
    }


def exact_grid_all() -> list[dict]:
    return json.loads((RESULTS / "leakage_grid_identity.json").read_text())["points"]


def fit_table(sims: dict, dev: dict, variants=("pooled", "slot")) -> dict:
    return {p: {v: loss(sim, dev, v) for v in variants} for p, sim in sims.items()}


def argmin(table: dict, variant: str, keys=None) -> tuple:
    keys = list(table) if keys is None else keys
    return min(keys, key=lambda p: table[p][variant][0])


# ------------------------------------------------------------------ cached exact simulations
@dataclass(frozen=True)
class Point:
    leak: float  # per CZ, every qubit
    strength: float  # P(partner of a leaked CZ gets an X error)
    spread: float  # P(partner of a leaked CZ leaks)
    life: float  # leaked lifetime, us
    read: float  # P(leaked qubit reads 1)

    def tag(self) -> str:
        return (
            f"L{self.leak:g}_S{self.strength:g}_T{self.spread:g}"
            f"_life{self.life:.4f}_read{self.read:.4f}"
        )


def points(life: float, read: float) -> dict:
    """The exact subset of the published grid at the given lifetime and leaked readout."""
    return {p: Point(*p, life, read) for p in EXACT_SUBSET}


_JOBS: dict = {}
_MODELS: dict = {}
_PROGS: dict = {}


def _job(name):
    if name not in _JOBS:
        _JOBS[name] = next(j for j in jobs() if j.name == name)
    return _JOBS[name]


def _model(name, basis, rounds):
    key = (name, basis, rounds)
    if key not in _MODELS:
        _MODELS[key] = _job(name).model(memory_index(basis, rounds))
    return _MODELS[key]


def _leakage(name, pt: Point):
    import frame_sim

    reset, readout = ibm_errors(_job(name).day)
    return frame_sim.Leakage(
        leak=AllQubits(defaultdict(lambda: pt.leak)),
        lifetime_us=pt.life,
        half_round_us=ROUND_US / 2,
        read_leaked=pt.read,
        partner_phase=pt.strength,
        partner_kind="bitflip",
        transfer=pt.spread,
        reset_error=reset,
        readout=readout,
        skipped_cz="dephase",
    )  # the field is not used by the exact simulators


def path_of(kind, name, basis, rounds, pt: Point, shots, seed, sim: str | None = None) -> Path:
    # inforce/: resets and readouts as ibm_errors gives them; older caches used the snapshot before
    sim = SIM if sim is None else sim
    folder = CACHE / (f"exact_{kind}" if sim == "t" else f"exact_{kind}_{sim}") / "inforce"
    return folder / f"{name}_{basis}{rounds}_{pt.tag()}_n{shots}_s{seed}.npz"


def _simulate(name, basis, rounds, pt: Point, shots, seed):
    model, _records, _labels = _model(name, basis, rounds)
    lk = _leakage(name, pt)
    if SIM == "t":
        import tableau_ref

        return tableau_ref.simulate(model, shots, lk, "identity", seed=seed)
    import exact_sim

    xl = exact_sim.Leakage(**{k: getattr(lk, k) for k in exact_sim.Leakage.__dataclass_fields__})
    key = (
        name,
        basis,
        rounds,
    )  # the program depends on the leakage only through the job's reset errors
    if key not in _PROGS:
        _PROGS[key] = exact_sim.Program(model, xl)
    return exact_sim.simulate(model, shots, xl, "identity", seed=seed, program=_PROGS[key])


def _run(task):
    kind, name, basis, rounds, pt, shots, seed = task
    out = path_of(kind, name, basis, rounds, pt, shots, seed)
    if out.exists():
        return str(out), 0.0
    out.parent.mkdir(parents=True, exist_ok=True)
    lock = out.with_name(out.name + ".lock")
    try:  # another script may be simulating the same task
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL))
    except FileExistsError:
        if time.time() - lock.stat().st_mtime < 3 * 3600:
            return str(out), -1.0
    t0 = time.time()
    model, _records, labels = _model(name, basis, rounds)
    det, obs = _simulate(name, basis, rounds, pt, shots, seed)
    part = out.with_name(out.name + ".part")
    with open(part, "wb") as fh:
        if kind == "det":
            np.savez_compressed(fh, det=np.packbits(det, axis=1), n=det.shape[1])
        elif kind == "fail":  # decoded as the reruns decode
            from heavyhex.experiment import run as R

            failed = R._decode(model, det) != obs
            patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
            np.savez_compressed(fh, failures=failed.sum(0), patches=np.array(patches), shots=shots)
        else:
            raise ValueError(kind)
    os.replace(part, out)
    lock.unlink(missing_ok=True)
    return str(out), time.time() - t0


def ensure(tasks, workers: int = WORKERS) -> None:
    """Run every (kind, job, basis, rounds, Point, shots, seed) task whose output is missing."""
    todo = [t for t in dict.fromkeys(tasks) if not path_of(*t).exists()]
    if not todo:
        return
    # longest first: 8-round, two-patch jobs
    todo.sort(key=lambda t: -(t[3] * t[5] * (2 if t[1].endswith("r1") else 1)))
    print(f"exact simulations: {len(todo)} runs on {workers} workers", flush=True)
    t0 = time.time()
    with mp.Pool(workers, maxtasksperchild=6) as pool:  # recycled workers keep memory bounded
        for i, (p, dt) in enumerate(pool.imap_unordered(_run, todo, chunksize=1), 1):
            if i % 8 == 0 or i == len(todo):
                print(
                    f"  {i}/{len(todo)} done ({time.time() - t0:.0f}s; last {dt:.0f}s {Path(p).name})",
                    flush=True,
                )
    # tasks another run had locked: wait for them
    while missing := [t for t in todo if not path_of(*t).exists()]:
        print(f"  waiting for {len(missing)} runs locked by another process", flush=True)
        time.sleep(60)


def load(name, basis, rounds, pt: Point, shots=GRID_SHOTS, seed=GRID_SEED, sim=None) -> np.ndarray:
    z = np.load(path_of("det", name, basis, rounds, pt, shots, seed, sim))
    return np.unpackbits(z["det"], axis=1, count=int(z["n"])).astype(bool)


def load_fail(name, basis, rounds, pt: Point, shots, seed) -> dict:
    z = np.load(path_of("fail", name, basis, rounds, pt, shots, seed))
    return {str(p): int(k) for p, k in zip(z["patches"], z["failures"])}
