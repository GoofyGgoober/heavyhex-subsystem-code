"""The six detection statistics of fit_leak2.py, pooled as published and round-conditioned.

stats(det, labels) returns fit_leak2.stats(det, labels) (checked equal on every call) plus

  co_data_slot, co_gauge_slot  the co-streak excess P(A&B | slot) - P(A|slot) P(B|slot)
                               computed in each of the 7 streak slots (rounds 2..8) and then
                               averaged over the slots, per pair, then over the pairs of the
                               class: the statistic of the corrected spatial diagnostic
                               (revision/costreak/costreak.py, excess_per_slot)
  repeat_slot                  the repeat excess P(fire r+1 | fired r) - P(fire r+1 | quiet r)
                               computed per round r and averaged over r (a further
                               sensitivity check; the review did not ask for it)
  co_unrel(_slot)              the same for unrelated same-patch pairs (not fitted)
  profile                      the mean streak probability in each slot

and loss(sim, dev, variant) is fit_leak2.py's loss (RMS of the six relative deviations),
with variant "pooled" (as published), "slot" (both co-streak statistics round-conditioned)
or "slot+repeat" (the repeat excess as well).
"""

from __future__ import annotations

import collections
import itertools

import numpy as np

import exact_cache  # noqa: F401  (sets sys.path)
from common import readout_series  # noqa: E402
from fit_leak2 import stats as stats_published  # noqa: E402
from where import data_read, z4_of  # noqa: E402

FIT_KEYS = ["X stab", "Z stab", "Z flag", "Z repeat", "Z co_data", "Z co_gauge"]
VARIANTS = {
    "pooled": {},
    "slot": {"Z co_data": "Z co_data_slot", "Z co_gauge": "Z co_gauge_slot"},
    "slot+repeat": {"Z co_data": "Z co_data_slot", "Z co_gauge": "Z co_gauge_slot", "Z repeat": "Z repeat_slot"},
}


def stats(det: np.ndarray, labels, check: bool = True) -> dict:
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
            per_r = ((now & nxt).sum(0) / np.maximum(now.sum(0), 1)
                     - (~now & nxt).sum(0) / np.maximum((~now).sum(0), 1))
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
        ref = stats_published(det, labels)
        for k, v in ref.items():
            assert (np.isnan(v) and np.isnan(out[k])) or abs(v - out[k]) <= 1e-12 * max(1.0, abs(v)), (k, v, out[k])
    return out


class Prepared:
    """stats() of one detector array under shot weights w (bootstrap counts), fast.

    Precomputes per-shot sums so that each statistic is a weighted sum over shots; with
    w = 1 it equals stats(det, labels), and with w = bincount(idx) it equals
    stats(det[idx], labels) (both checked in selftest()).
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
        self.pairs = {k: (np.array([p[0] for p in v]), np.array([p[1] for p in v])) for k, v in cls.items()}

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
        M = np.stack([((self.st[:, t, :] * wf[:, None]).T @ self.st[:, t, :]).astype(np.float64) for t in range(self.T)])
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
            out[cls + "_slot"] = float(np.mean((pj_slot[:, i, j] - p_slot[:, i] * p_slot[:, j]).mean(0)))
        out["profile"] = p_slot.mean(1).tolist()
        return out


def selftest(det: np.ndarray, labels, seed: int = 0) -> float:
    """Largest relative difference between Prepared and stats(), plain and resampled."""
    pr = Prepared(det, labels)
    idx = np.random.default_rng(seed).integers(0, len(det), len(det))
    worst = 0.0
    for a, b in ((pr.stats(), stats(det, labels)), (pr.stats(np.bincount(idx, minlength=len(det))), stats(det[idx], labels))):
        for k in b:
            if k == "profile":
                d = np.max(np.abs(np.array(a[k]) - np.array(b[k])))
            elif np.isnan(b[k]):
                d = 0.0 if np.isnan(a[k]) else np.inf
            else:
                d = abs(a[k] - b[k]) / max(abs(b[k]), 1e-6)
            worst = max(worst, d)
    return worst


def merge(per_case: list[dict]) -> dict:
    """Mean over the cases of each '<basis> <stat>' (nanmean, as fit_leak2.py)."""
    tot = collections.defaultdict(list)
    for s in per_case:
        for k, v in s.items():
            tot[k].append(v)
    return {k: (np.mean(v, axis=0).tolist() if k.endswith("profile") else float(np.nanmean(v)))
            for k, v in tot.items()}


def loss(sim: dict, dev: dict, variant: str = "pooled") -> tuple[float, dict]:
    sub = VARIANTS[variant]
    rel = {}
    for k in FIT_KEYS:
        kk = sub.get(k, k)
        rel[k] = (sim[kk] - dev[kk]) / abs(dev[kk])
    return float(np.sqrt(np.mean([v ** 2 for v in rel.values()]))), rel
