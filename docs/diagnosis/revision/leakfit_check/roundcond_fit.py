"""Leakage grid fit with a round-conditioned co-streak statistic, under the exact simulator.

Review items 10 and 13. fit_leak2.py's co-streak statistics pool the 7 streak slots of a
shot before subtracting P(A) P(B), which adds the covariance over slots of the two streak
probabilities (both rise with the round). The round-conditioned statistic computes the
excess per slot and then averages over the slots (the corrected spatial diagnostic's
statistic, revision/costreak). The fit replaces the two pooled co-streak statistics by
these, for device and simulation alike; the other four statistics are unchanged.

Simulations: not rerun here. revision/leakage_exact/grid_exact.py (the exact-model grid
fit) computes both versions from the same simulated shots at each grid point (exact
simulation, CZ with a leaked operand does not act; 8000 shots per circuit; seed 21) and
stores them in leakage_exact/grid_identity.json. This script reads that file and adds:

  - the loss table and the argmin under each statistic, on the published grid and on all
    points present (leakage_exact's extended grid, if run);
  - the device statistics recomputed here from the run folders (fitstats.py), checked
    against leakage_exact/device_stats.json and results/fit_leak2_bitflip.json;
  - a device-shot bootstrap (B replicates resampling the 2500 shots of each of the eight
    circuits; simulated statistics held fixed): how often each point wins, under each
    statistic, and how often the round-conditioned argmin equals the pooled one;
  - Monte Carlo spread of the loss: the published grid's three best points under each
    statistic simulated again here with seeds 31 and 32 (exact_cache.py, the same exact simulator), and any
    point leakage_exact ran with several seeds.

The Z-memory gap closure depends only on the chosen point: if the argmin does not move,
the closure is that of the exact rerun at the published optimum (leaky_fix/
rerun_exactidentity_paperparams.json, or leakage_exact's rerun when present).

    python docs/diagnosis/revision/leakfit_check/roundcond_fit.py [--boot 500]

Writes roundcond_fit.json; the log is roundcond_fit.log.
"""

from __future__ import annotations

import argparse
import collections
import itertools
import json
import sys
from pathlib import Path

import numpy as np

import exact_cache as E
from exact_cache import FIT_JOBS, Point

HERE = Path(__file__).resolve().parent
LEAK_EXACT = E.REV / "leakage_exact"
PAPER_GRID = list(itertools.product((0.001, 0.002, 0.003), (0.25, 0.5, 1.0), (0.0, 0.05)))
BEST_PUBLISHED = (0.002, 0.25, 0.05)
MC_SEEDS = (31, 32)
CASES = [(j, b) for j in FIT_JOBS for b in "XZ"]
FIT_VARIANTS = ("pooled", "slot")


def subset():
    from fit_leak2_fix import EXACT_SUBSET
    return list(EXACT_SUBSET)


def points(life: float, read: float, with_profile: bool = False) -> dict:
    """The exact subset of the published grid (fit_leak2_fix.EXACT_SUBSET) at the given inputs."""
    assert not with_profile
    return {p: Point(*p, life, read) for p in subset()}


def exact_grid(seed: int = 21, shots: int = 8000) -> dict:
    """{(leak, strength, spread): sim statistics} from leakage_exact/grid_identity.json."""
    d = json.loads((LEAK_EXACT / "grid_identity.json").read_text())
    return {(q["leak"], q["strength"], q["spread"]): q["sim"] for q in d["points"]
            if q["seed"] == seed and q["shots"] == shots}


def exact_grid_all() -> list[dict]:
    return json.loads((LEAK_EXACT / "grid_identity.json").read_text())["points"]


def device_cases():
    from common import jobs, memory_index
    out = {}
    for j in [j for j in jobs() if j.name in FIT_JOBS]:
        for b in "XZ":
            model, records, labels = j.model(memory_index(b, 8))
            chip, _ = j.chip_detectors(memory_index(b, 8), model, records)
            out[(j.name, b)] = (chip, labels)
    return out


def merged(per_case: dict) -> dict:
    import fitstats as S
    return S.merge({f"{b} {k}": v for k, v in per_case[(j, b)].items()} for j, b in CASES)


def case_stats(arrays: dict, labels: dict, check=False) -> dict:
    import fitstats as S
    return {c: S.stats(arrays[c], labels[c], check=check) for c in CASES}


def fit_table(sims: dict, dev: dict, variants=FIT_VARIANTS) -> dict:
    import fitstats as S
    return {p: {v: S.loss(sim, dev, v) for v in variants} for p, sim in sims.items()}


def argmin(table: dict, variant: str, keys=None) -> tuple:
    keys = list(table) if keys is None else keys
    return min(keys, key=lambda p: table[p][variant][0])


def boot_weights(sizes: dict, B: int, seed: int) -> dict:
    """{case: B x N shot counts}: resampling shots with replacement within each case."""
    rng = np.random.default_rng(seed)
    return {c: np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(np.int16)
            for c, n in sizes.items()}


def boot_stats(arrays: dict, labels: dict, weights: dict) -> list[dict]:
    """Merged statistics of each bootstrap replicate (fitstats.Prepared, one case at a time)."""
    import fitstats as S
    B = len(next(iter(weights.values())))
    per = [dict() for _ in range(B)]
    for c in CASES:
        pr = S.Prepared(arrays[c], labels[c])
        for r in range(B):
            per[r][c] = pr.stats(weights[c][r])
        del pr
    return [merged(x) for x in per]


def main() -> None:
    import fitstats as S
    ap = argparse.ArgumentParser()
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    dev_cases = device_cases()
    labels = {c: dev_cases[c][1] for c in CASES}
    dev_arr = {c: dev_cases[c][0] for c in CASES}
    dev = merged(case_stats(dev_arr, labels, check=True))
    pub = json.loads((E.DIAG / "results/fit_leak2_bitflip.json").read_text())["chip"]
    le_dev = json.loads((LEAK_EXACT / "device_stats.json").read_text())
    d1 = max(abs(dev[k] - pub[k]) for k in pub)
    d2 = max(abs(dev[k] - le_dev[k]) for k in le_dev)
    print(f"device statistics: equal to results/fit_leak2_bitflip.json to {d1:.1e}, to leakage_exact/device_stats.json to {d2:.1e}")
    print("device: " + "  ".join(f"{k} {dev[k]:.5g}" for k in S.FIT_KEYS) +
          f"  | per slot: Z co_data {dev['Z co_data_slot']:.5g}  Z co_gauge {dev['Z co_gauge_slot']:.5g}")
    print("device streak probability by slot: " + " ".join(f"{x:.4f}" for x in dev["Z profile"]))
    for k in ("co_data", "co_gauge"):
        print(f"  {k}: pooled - per slot = {1e4*(dev['Z '+k]-dev['Z '+k+'_slot']):.2f}e-4 "
              f"({100*(dev['Z '+k]-dev['Z '+k+'_slot'])/dev['Z '+k]:.1f}% of pooled)")

    sims = exact_grid()
    table = fit_table(sims, dev)
    paper = [p for p in PAPER_GRID if p in sims]
    allp = list(sims)
    print(f"\nleakage_exact/grid_identity.json: {len(allp)} points at seed 21, 8000 shots ({len(paper)} of the 18 published-grid points)")
    # consistency with leaky_fix's exact-identity subset (other implementation, other random numbers)
    ref = json.loads((E.FIX / "fit_leak2_bitflip_exact-identity_paper.json").read_text())
    cmp_ = [(tuple((r["leak"], r["strength"], r["spread"])), r["loss"]) for r in ref["points"]]
    print("pooled loss vs leaky_fix/tableau_ref exact subset (independent random numbers): " +
          ", ".join(f"{p} {table[p]['pooled'][0]:.3f}/{l:.3f}" for p, l in cmp_ if p in table))

    print("\nloss by point, sorted by the pooled loss (co-streak x1e-4: pooled / per slot, simulated)")
    for p in sorted(allp, key=lambda q: table[q]["pooled"][0]):
        sim = sims[p]
        print(f"  {str(p):24s} pooled {table[p]['pooled'][0]:.3f}  slot {table[p]['slot'][0]:.3f}"
              f"   co_data {1e4*sim['Z co_data']:.2f}/{1e4*sim['Z co_data_slot']:.2f}"
              f"  co_gauge {1e4*sim['Z co_gauge']:.2f}/{1e4*sim['Z co_gauge_slot']:.2f}")
    best = {}
    keysets = {"published grid": paper, "all points": allp}
    for v in FIT_VARIANTS:
        for n, keys in keysets.items():
            if not keys:
                continue
            p = argmin(table, v, keys)
            ranked = sorted(keys, key=lambda q: table[q][v][0])
            best[f"{v} {n}"] = p
            print(f"best ({v}, {n}): {p} loss {table[p][v][0]:.3f}; runner-up {ranked[1]} {table[ranked[1]][v][0]:.3f}  | residuals " +
                  "  ".join(f"{k} {100*x:+.0f}%" for k, x in table[p][v][1].items()))

    # Monte Carlo spread where several seeds exist
    reps = collections.defaultdict(list)
    for q in exact_grid_all():
        if q["shots"] == 8000:
            reps[(q["leak"], q["strength"], q["spread"])].append(q)
    mc = {}
    for p, qs in reps.items():
        if len(qs) > 1:
            lp = [S.loss(q["sim"], dev, "pooled")[0] for q in qs]
            ls = [S.loss(q["sim"], dev, "slot")[0] for q in qs]
            mc[str(p)] = {"seeds": [q["seed"] for q in qs], "pooled": lp, "slot": ls}
            print(f"Monte Carlo: {p} over seeds {[q['seed'] for q in qs]}: pooled {np.mean(lp):.3f} +- {np.std(lp, ddof=1):.3f}, "
                  f"slot {np.mean(ls):.3f} +- {np.std(ls, ddof=1):.3f}")

    # Monte Carlo replicates of the leading points: two more seeds each (exact_cache, same simulator)
    life, read = E.hmm_inputs()
    lead = []  # the published grid's three best points under each statistic
    for v in FIT_VARIANTS:
        lead += sorted(paper, key=lambda q: table[q][v][0])[:3]
    lead = list(dict.fromkeys(lead))
    E.ensure([("det", j, b, 8, Point(*p, life, read), E.GRID_SHOTS, s) for p in lead for s in MC_SEEDS for j, b in CASES], a.workers)
    mc_rep = {}
    for p in lead:
        sims_p = [sims[p]] + [merged(case_stats({c: E.load(*c, 8, Point(*p, life, read), E.GRID_SHOTS, s) for c in CASES}, labels))
                              for s in MC_SEEDS]
        mc_rep[p] = {v: [S.loss(x, dev, v)[0] for x in sims_p] for v in FIT_VARIANTS}
        mc_rep[p]["sims"] = sims_p
    print(f"\nMonte Carlo replicates (seed 21 from leakage_exact, seeds {MC_SEEDS} here), loss mean +- sd over the 3:")
    for p in lead:
        print(f"  {str(p):24s} pooled {np.mean(mc_rep[p]['pooled']):.3f} +- {np.std(mc_rep[p]['pooled'], ddof=1):.3f}"
              f"   slot {np.mean(mc_rep[p]['slot']):.3f} +- {np.std(mc_rep[p]['slot'], ddof=1):.3f}"
              f"   ({', '.join(f'{x:.3f}' for x in mc_rep[p]['pooled'])} | {', '.join(f'{x:.3f}' for x in mc_rep[p]['slot'])})")
    for v in FIT_VARIANTS:
        per_seed = [min(lead, key=lambda q: mc_rep[q][v][i]) for i in range(1 + len(MC_SEEDS))]
        avg = min(lead, key=lambda q: np.mean(mc_rep[q][v]))
        print(f"  {v}: best among these per seed {per_seed}; best on the 3-seed mean {avg}")
        best[f"{v} 3-seed mean (leading points)"] = avg

    # device-shot bootstrap of the argmin (simulated statistics fixed)
    print(f"\ndevice-shot bootstrap, {a.boot} replicates (simulated statistics fixed)", flush=True)
    wd = boot_weights({c: len(dev_arr[c]) for c in CASES}, a.boot, 2026)
    dev_reps = boot_stats(dev_arr, labels, wd)
    wins = {f"{v} {n}": collections.Counter() for v in FIT_VARIANTS for n in keysets if keysets[n]}
    same = collections.Counter()
    loss_best = {v: [] for v in FIT_VARIANTS}
    for d in dev_reps:
        tb = fit_table(sims, d)
        for n, keys in keysets.items():
            if not keys:
                continue
            am = {v: argmin(tb, v, keys) for v in FIT_VARIANTS}
            for v in FIT_VARIANTS:
                wins[f"{v} {n}"][am[v]] += 1
            same[n] += am["pooled"] == am["slot"]
        for v in FIT_VARIANTS:
            loss_best[v].append(tb[BEST_PUBLISHED][v][0])
    for k, c in wins.items():
        print(f"  {k:24s} " + ", ".join(f"{p} {n/a.boot:.0%}" for p, n in c.most_common(5)))
    for n in same:
        print(f"  {n}: the round-conditioned argmin equals the pooled one in {same[n]/a.boot:.0%} of replicates")
    for v in FIT_VARIANTS:
        print(f"  {v}: loss at {BEST_PUBLISHED}, 95% bootstrap range {np.percentile(loss_best[v], 2.5):.3f}-{np.percentile(loss_best[v], 97.5):.3f}")
    se = {k: float(np.std([d[k] for d in dev_reps], ddof=1) / abs(dev[k])) for k in S.FIT_KEYS + ["Z co_data_slot", "Z co_gauge_slot"]}
    print("  device relative standard errors: " + "  ".join(f"{k} {100*v:.1f}%" for k, v in se.items()))

    out = {"device": dev, "device_rel_se": se, "grid_source": "revision/leakage_exact/grid_identity.json (seed 21, 8000 shots)",
           "points": [{"point": list(p), "loss_pooled": table[p]["pooled"][0], "loss_slot": table[p]["slot"][0],
                       "rel_pooled": table[p]["pooled"][1], "rel_slot": table[p]["slot"][1],
                       "co_data": sims[p]["Z co_data"], "co_data_slot": sims[p]["Z co_data_slot"],
                       "co_gauge": sims[p]["Z co_gauge"], "co_gauge_slot": sims[p]["Z co_gauge_slot"]} for p in allp],
           "best": {k: list(v) for k, v in best.items()}, "monte_carlo": mc,
           "monte_carlo_leading": {str(p): {v: mc_rep[p][v] for v in FIT_VARIANTS} for p in lead},
           "bootstrap": {"B": a.boot, "wins": {k: {str(p): n for p, n in c.items()} for k, c in wins.items()},
                         "same_argmin_share": {n: same[n] / a.boot for n in same},
                         "loss_at_published_optimum_95": {v: np.percentile(loss_best[v], [2.5, 97.5]).tolist() for v in FIT_VARIANTS}}}
    (HERE / "roundcond_fit.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("\nwrote", HERE / "roundcond_fit.json")


if __name__ == "__main__":
    main()
