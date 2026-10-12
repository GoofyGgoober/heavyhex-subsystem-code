"""Co-streak excess of pairs of flags and relays, with errors clustered by physical pair.

Do two flags (or relays) that read a common data qubit fire in streaks together? If the data
qubits leak, they should; if only the readout qubits leak, their streaks are independent.
The model shots are 20,000 per job from stim with seed 4. Standard errors allow for the same
physical pair of qubits recurring across jobs and patches.

Definitions:
  A flag or relay series is its detector bit in each of the 8 rounds of a shot. It is "in a
  streak" at round r (r = 2..8) if it fires at r - 1 and at r; so each shot gives 7 streak
  slots. For two series A and B recorded in the same job, the co-streak excess is
      P(A and B both in a streak) - P(A in a streak) P(B in a streak),
  where each P is the fraction of the job's (shot, slot) cells, pooled over the 2,500 shots
  and the 7 slots. It is zero for independent series. One value per (job, pair).
  Pair classes: "share a data qubit" (the two read a common data qubit through their CXs),
  "same Z gauge, no shared data" (the two flags of one bulk Z4 gauge, sharing only its
  ancilla), "same patch, unrelated", and "different patches" (one series in each patch of a
  repetition-1 job). The model value is the same mean over the same (job, pair) rows,
  computed on the calibration model's shots.

Standard errors of the class mean (mean over its n (job, pair) rows):
  plain      sd / sqrt(n), rows treated as independent;
  pair       cluster-robust (CR1), clusters = the unordered pair of physical chip qubits
             {q_A, q_B}: se^2 = G/(G-1) * sum_g (sum_{i in g} (y_i - ybar))^2 / n^2,
             G the number of distinct physical pairs in the class;
  qubit      dyadic-robust, a sensitivity check: rows i and j may be correlated whenever
             their pairs share at least one physical qubit (not only the same pair):
             se^2 = sum_{i,j share a qubit} e_i e_j / n^2 (no small-sample factor).
Also reported:
  - the mean over physical pairs of each pair's mean (every physical pair weighted
    equally) with sd/sqrt(G);
  - model_exact: the model's value without sampling noise, from its detector error model
    (class ExactModel). stim's seeded sampling differs between machines and builds, so the
    sampled model values differ between machines by sampling noise only;
  - the mean streak probability in each of the 7 slots, chip and model;
  - as a definition check, the excess computed per streak slot and then averaged over the
    7 slots. Pooling the slots adds the covariance over slots of the two series' streak
    probabilities, which is not zero when both rise from round to round.

A few minutes. Writes results/costreak.json, and cache/costreak_pairs.csv (one row per job
and pair), and prints the tables.

    python docs/analysis/costreak.py
"""

from __future__ import annotations

import csv
import itertools
import json
from collections import defaultdict

import numpy as np

from common import CACHE, RESULTS, data_read, jobs, memory_index, readout_series, streaks, z4_of

KINDS = [
    "share a data qubit",
    "same Z gauge, no shared data",
    "same patch, unrelated",
    "different patches",
]
SEED, SHOTS = 4, 20000
PARITY = np.array([bin(i).count("1") & 1 for i in range(16)], dtype=bool)


def kind_of(a, b) -> str:
    """The pair's class, tested in this order."""
    if a[0] != b[0]:
        return "different patches"
    if data_read(*a) & data_read(*b):
        return "share a data qubit"
    if z4_of(a[0], a[1]) and z4_of(a[0], a[1]) == z4_of(b[0], b[1]):
        return "same Z gauge, no shared data"
    return "same patch, unrelated"


def excess_pooled(sa: np.ndarray, sb: np.ndarray) -> float:
    """The pooled statistic: over shots and slots."""
    return float((sa & sb).mean() - sa.mean() * sb.mean())


def excess_per_slot(sa: np.ndarray, sb: np.ndarray) -> float:
    """The same per streak slot, then averaged over the slots."""
    return float(np.mean((sa & sb).mean(0) - sa.mean(0) * sb.mean(0)))


class ExactModel:
    """Exact detector-firing probabilities of the calibration model, without sampling.

    Uses the model's independent error mechanisms as stim lists them (the same detector
    error model as common.marginals): the probability that a set of detectors all fire
    is (1/2^n) sum_S (-1)^|S| prod_k (1 - 2 p_k)^[mechanism k flips an odd number of S].
    """

    def __init__(self, model) -> None:
        dem = model.detector_error_model(decompose_errors=False, approximate_disjoint_errors=True)
        self.p, self.by_det = [], defaultdict(list)
        for ins in dem.flattened():
            if ins.type != "error":
                continue
            k = len(self.p)
            self.p.append(ins.args_copy()[0])
            for t in ins.targets_copy():
                if t.is_relative_detector_id():
                    self.by_det[t.val].append(k)
        self.p = np.array(self.p)

    def all_fire(self, dets: list[int]) -> float:
        n = len(dets)
        mechs = sorted(set().union(*(self.by_det[d] for d in dets)))
        mask = np.zeros(len(mechs), dtype=np.int64)
        pos = {m: j for j, m in enumerate(mechs)}
        for i, d in enumerate(dets):
            for m in self.by_det[d]:
                mask[pos[m]] ^= 1 << i
        f = 1 - 2 * self.p[mechs]
        total = 0.0
        for S in range(1 << n):
            odd = PARITY[mask & S]
            total += (-1) ** int(PARITY[S]) * float(np.prod(f[odd]))
        return total / (1 << n)


def collect() -> tuple[list[dict], dict]:
    idx = memory_index("Z", 8)
    rows = []
    by_slot = {"chip": [], "model": [], "model_exact": []}
    for job in jobs():
        model, records, labels = job.model(idx)
        chip, _ = job.chip_detectors(idx, model, records)
        sim = model.compile_detector_sampler(seed=SEED).sample(SHOTS)
        exact = ExactModel(model)
        series = readout_series(labels)
        keys = list(series)
        qubit = {k: job.readout_qubit(*k) for k in keys}
        if len(set(qubit.values())) != len(keys):
            raise RuntimeError(f"{job.name}: two series on one physical qubit")
        st = {
            src: {k: streaks(x[:, series[k]]) for k in keys}
            for src, x in (("chip", chip), ("model", sim))
        }
        # exact streak probability of each series in each slot (fires at r-1 and r)
        nslot = len(series[keys[0]]) - 1
        pe = {
            k: np.array([exact.all_fire([series[k][r], series[k][r + 1]]) for r in range(nslot)])
            for k in keys
        }
        for k in keys:
            by_slot["chip"].append(st["chip"][k].mean(0))
            by_slot["model"].append(st["model"][k].mean(0))
            by_slot["model_exact"].append(pe[k])
        for a, b in itertools.combinations(keys, 2):
            qa, qb = sorted((qubit[a], qubit[b]))
            kind = kind_of(a, b)
            row = {
                "job": job.name,
                "day": job.day,
                "rep": job.rep,
                "kind": kind,
                "a": f"{a[0]}|{a[1]}|{a[2]}",
                "b": f"{b[0]}|{b[1]}|{b[2]}",
                "qa": qa,
                "qb": qb,
                "shots": int(chip.shape[0]),
                "slots": int(st["chip"][a].shape[1]),
            }
            for src in ("chip", "model"):
                sa, sb = st[src][a], st[src][b]
                row[src] = excess_pooled(sa, sb)
                row[src + "_per_slot"] = excess_per_slot(sa, sb)
                row[src + "_pa"], row[src + "_pb"], row[src + "_pab"] = (
                    float(sa.mean()),
                    float(sb.mean()),
                    float((sa & sb).mean()),
                )
            # if no error mechanism of the model flips a detector of each series, the joint
            # probability is the product; otherwise sum over the four detectors of each slot
            ma = set().union(*(exact.by_det[d] for d in series[a]))
            mb = set().union(*(exact.by_det[d] for d in series[b]))
            shared = bool(ma & mb)
            if shared:
                pab = np.array(
                    [
                        exact.all_fire(
                            [series[a][r], series[a][r + 1], series[b][r], series[b][r + 1]]
                        )
                        for r in range(nslot)
                    ]
                )
            else:
                pab = pe[a] * pe[b]
            row["model_exact"] = float(pab.mean() - pe[a].mean() * pe[b].mean())
            row["model_exact_per_slot"] = float(np.mean(pab - pe[a] * pe[b]))
            row["model_exact_pa"], row["model_exact_pb"] = float(pe[a].mean()), float(pe[b].mean())
            row["shares_mechanism"] = shared
            rows.append(row)
        print(job.name, "done", flush=True)
    slots = {src: [float(v) for v in np.mean(by_slot[src], axis=0)] for src in by_slot}
    slots["series"] = len(by_slot["chip"])
    return rows, slots


def se_plain(y: np.ndarray) -> float:
    return float(y.std(ddof=1) / np.sqrt(len(y)))


def se_cluster(y: np.ndarray, groups: list) -> tuple[float, int]:
    """CR1 standard error of the mean, clustered by groups."""
    e = y - y.mean()
    sums = defaultdict(float)
    for g, v in zip(groups, e):
        sums[g] += v
    G = len(sums)
    s = np.array(list(sums.values()))
    return float(np.sqrt(G / (G - 1) * (s**2).sum()) / len(y)), G


def se_dyadic(y: np.ndarray, qa: np.ndarray, qb: np.ndarray) -> float:
    """Rows correlated whenever their physical pairs share a qubit."""
    e = y - y.mean()
    share = (
        (qa[:, None] == qa[None, :])
        | (qa[:, None] == qb[None, :])
        | (qb[:, None] == qa[None, :])
        | (qb[:, None] == qb[None, :])
    )
    return float(np.sqrt(max(e @ share @ e, 0.0)) / len(y))


def pair_means(y: np.ndarray, groups: list) -> np.ndarray:
    acc = defaultdict(list)
    for g, v in zip(groups, y):
        acc[g].append(v)
    return np.array([np.mean(v) for v in acc.values()])


def summarize(rows: list[dict]) -> dict:
    out = {}
    for kind in KINDS:
        rs = [r for r in rows if r["kind"] == kind]
        c = np.array([r["chip"] for r in rs])
        m = np.array([r["model"] for r in rs])
        qa = np.array([r["qa"] for r in rs])
        qb = np.array([r["qb"] for r in rs])
        groups = [(r["qa"], r["qb"]) for r in rs]
        se_pair, G = se_cluster(c, groups)
        se_pair_model, _ = se_cluster(m, groups)
        se_pair_diff, _ = se_cluster(c - m, groups)
        pm = pair_means(c, groups)
        per_pair_count = np.bincount(np.unique(groups, axis=0, return_inverse=True)[1].ravel())
        cs = np.array([r["chip_per_slot"] for r in rs])
        me = np.array([r["model_exact"] for r in rs])
        mes = np.array([r["model_exact_per_slot"] for r in rs])
        by_job = [r["job"] for r in rs]
        # sampled model against the exact one: the sampling noise is shared by pairs of one job
        mc_dev, mc_dev_se = float((m - me).mean()), se_cluster(m - me, by_job)[0]
        out[kind] = {
            "rows": len(rs),
            "physical_pairs": G,
            "physical_qubits": len(set(qa) | set(qb)),
            "jobs": len({r["job"] for r in rs}),
            "rows_per_physical_pair": {
                "min": int(per_pair_count.min()),
                "median": float(np.median(per_pair_count)),
                "max": int(per_pair_count.max()),
            },
            "chip": float(c.mean()),
            "chip_se_plain": se_plain(c),
            "chip_se_pair_clustered": se_pair,
            "chip_se_qubit_dyadic": se_dyadic(c, qa, qb),
            "model": float(m.mean()),
            "model_se_plain": se_plain(m),
            "model_se_pair_clustered": se_pair_model,
            "chip_minus_model": float((c - m).mean()),
            "chip_minus_model_se_pair_clustered": se_pair_diff,
            "chip_mean_of_pair_means": float(pm.mean()),
            "chip_mean_of_pair_means_se": float(pm.std(ddof=1) / np.sqrt(len(pm))),
            "model_exact": float(me.mean()),
            "model_sampled_minus_exact": mc_dev,
            "model_sampled_minus_exact_se_job_clustered": mc_dev_se,
            "rows_sharing_an_error_mechanism_in_model": int(sum(r["shares_mechanism"] for r in rs)),
            "chip_minus_model_exact": float((c - me).mean()),
            "chip_minus_model_exact_se_pair_clustered": se_cluster(c - me, groups)[0],
            "per_slot_definition": {
                "chip": float(cs.mean()),
                "chip_se_plain": se_plain(cs),
                "chip_se_pair_clustered": se_cluster(cs, groups)[0],
                "chip_se_qubit_dyadic": se_dyadic(cs, qa, qb),
                "model": float(np.mean([r["model_per_slot"] for r in rs])),
                "model_exact": float(mes.mean()),
            },
            "chip_streak_probability_mean": float(
                np.mean([(r["chip_pa"] + r["chip_pb"]) / 2 for r in rs])
            ),
            "model_streak_probability_mean": float(
                np.mean([(r["model_pa"] + r["model_pb"]) / 2 for r in rs])
            ),
            "model_exact_streak_probability_mean": float(
                np.mean([(r["model_exact_pa"] + r["model_exact_pb"]) / 2 for r in rs])
            ),
        }
    return out


def main() -> None:
    rows, slots = collect()
    CACHE.mkdir(exist_ok=True)
    with open(CACHE / "costreak_pairs.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    out = summarize(rows)
    result = {
        "script": "docs/analysis/costreak.py",
        "circuit": "8-round Z memory, all 15 jobs, 2500 shots each; model: stim, 20000 shots per job, seed 4",
        "definition": (
            "co-streak excess = P(both in a streak) - P(A in a streak) P(B in a streak); in a streak at "
            "round r = fires at r-1 and r (r = 2..8, 7 slots); P pooled over shots and slots of one job; "
            "class value = mean over (job, pair) rows"
        ),
        "note_model_sampling": (
            "stim's seeded sampling is not reproducible across machines and builds, so the "
            "sampled model values differ between machines by sampling noise; model_exact "
            "removes that noise"
        ),
        "streak_probability_by_slot": slots,
        "classes": out,
    }
    (RESULTS / "costreak.json").write_text(json.dumps(result, indent=1) + "\n")
    print("\nstreak probability by slot (mean over", slots["series"], "series):")
    for src in ("chip", "model", "model_exact"):
        print(f"  {src:12s}", " ".join(f"{v:.4f}" for v in slots[src]))
    print(
        f"\npooled over slots, units 1e-4\n{'class':30s} {'rows':>5s} {'pairs':>5s}  {'chip':>6s} "
        f"{'pair-cl':>7s} {'plain':>6s} {'dyadic':>6s}  {'model':>7s} {'exact':>7s}  {'pair-means':>11s}"
    )
    for kind in KINDS:
        o = out[kind]
        print(
            f"{kind:30s} {o['rows']:5d} {o['physical_pairs']:5d}  {1e4 * o['chip']:6.2f} "
            f"{1e4 * o['chip_se_pair_clustered']:7.2f} {1e4 * o['chip_se_plain']:6.2f} "
            f"{1e4 * o['chip_se_qubit_dyadic']:6.2f}  {1e4 * o['model']:7.3f} {1e4 * o['model_exact']:7.3f}  "
            f"{1e4 * o['chip_mean_of_pair_means']:5.2f}+-{1e4 * o['chip_mean_of_pair_means_se']:.2f}"
        )
    print(
        f"\nper slot, then averaged over slots, units 1e-4\n{'class':30s}  {'chip':>6s} {'pair-cl':>7s} "
        f"{'plain':>6s} {'dyadic':>6s}  {'model':>7s} {'exact':>7s}"
    )
    for kind in KINDS:
        o = out[kind]["per_slot_definition"]
        print(
            f"{kind:30s}  {1e4 * o['chip']:6.2f} {1e4 * o['chip_se_pair_clustered']:7.2f} {1e4 * o['chip_se_plain']:6.2f} "
            f"{1e4 * o['chip_se_qubit_dyadic']:6.2f}  {1e4 * o['model']:7.3f} {1e4 * o['model_exact']:7.3f}"
        )


if __name__ == "__main__":
    main()
