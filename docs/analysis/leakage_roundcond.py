"""The leakage grid fit with a round-conditioned co-streak statistic, under the exact simulator.

The fit's co-streak statistics pool the 7 streak slots of a shot before subtracting
P(A) P(B), which adds the covariance over slots of the two streak probabilities (both rise
with the round). The round-conditioned statistic computes the excess per slot and then
averages over the slots (as costreak.py does). This fit replaces the two pooled co-streak
statistics by these, for device and simulation alike; the other four are unchanged.

The grid is not simulated again: leakage_grid.py computes both versions from the same
shots at each point (results/leakage_grid_identity.json). This adds:

  - the loss table and the argmin under each statistic, on the published grid and on all
    points present;
  - the device statistics recomputed from the run folders, checked against
    results/leakage_grid_device.json and results/leakage_frame_grid_bitflip.json;
  - a device-shot bootstrap (B replicates resampling the 2500 shots of each of the eight
    circuits; simulated statistics held fixed): how often each point wins, under each
    statistic, and how often the round-conditioned argmin equals the pooled one;
  - the Monte Carlo spread of the loss: the published grid's three best points under each
    statistic simulated again with seeds 31 and 32 (leakage.ensure, cached), and any point
    the grid ran with several seeds.

The Z-memory gap closure depends only on the chosen point: if the argmin does not move,
the closure is that of the exact rerun at the published optimum (leakage_fit.py).
Moderate from the cache; the seed replicates are heavy the first time.

    python docs/analysis/leakage_roundcond.py [--boot 500] [--workers 2]

Writes results/leakage_roundcond.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

import numpy as np

import leakage as E
from common import RESULTS, WORKERS
from leakage import (
    BEST_PUBLISHED,
    CASES,
    PAPER_GRID,
    Point,
    argmin,
    case_stats,
    device_cases,
    exact_grid,
    exact_grid_all,
    fit_table,
    merged,
)

MC_SEEDS = (31, 32)
FIT_VARIANTS = ("pooled", "slot")


def boot_weights(sizes: dict, B: int, seed: int) -> dict:
    """{case: B x N shot counts}: resampling shots with replacement within each case."""
    rng = np.random.default_rng(seed)
    return {
        c: np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(B)]).astype(
            np.int16
        )
        for c, n in sizes.items()
    }


def boot_stats(arrays: dict, labels: dict, weights: dict) -> list[dict]:
    """Merged statistics of each bootstrap replicate (leakage.Prepared, one case at a time)."""
    B = len(next(iter(weights.values())))
    per = [{} for _ in range(B)]
    for c in CASES:
        pr = E.Prepared(arrays[c], labels[c])
        for r in range(B):
            per[r][c] = pr.stats(weights[c][r])
        del pr
    return [merged(x) for x in per]


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")

    dev_cases = device_cases()
    labels = {c: dev_cases[c][1] for c in CASES}
    dev_arr = {c: dev_cases[c][0] for c in CASES}
    dev = merged(case_stats(dev_arr, labels, check=True))
    pub = json.loads((RESULTS / "leakage_frame_grid_bitflip.json").read_text())["chip"]
    le_dev = json.loads((RESULTS / "leakage_grid_device.json").read_text())
    d1 = max(abs(dev[k] - pub[k]) for k in pub)
    d2 = max(abs(dev[k] - le_dev[k]) for k in le_dev)
    print(
        f"device statistics: equal to leakage_frame_grid_bitflip.json to {d1:.1e}, to leakage_grid_device.json to {d2:.1e}"
    )
    print(
        "device: "
        + "  ".join(f"{k} {dev[k]:.5g}" for k in E.FIT_KEYS)
        + f"  | per slot: Z co_data {dev['Z co_data_slot']:.5g}  Z co_gauge {dev['Z co_gauge_slot']:.5g}"
    )
    print("device streak probability by slot: " + " ".join(f"{x:.4f}" for x in dev["Z profile"]))
    for k in ("co_data", "co_gauge"):
        print(
            f"  {k}: pooled - per slot = {1e4 * (dev['Z ' + k] - dev['Z ' + k + '_slot']):.2f}e-4 "
            f"({100 * (dev['Z ' + k] - dev['Z ' + k + '_slot']) / dev['Z ' + k]:.1f}% of pooled)"
        )

    sims = exact_grid()
    table = fit_table(sims, dev)
    paper = [p for p in PAPER_GRID if p in sims]
    allp = list(sims)
    print(
        f"\nleakage_grid_identity.json: {len(allp)} points at seed 21, 8000 shots ({len(paper)} of the 18 published-grid points)"
    )
    # consistency with tableau_ref.py's exact subset (other implementation, other random numbers)
    ref = json.loads((RESULTS / "leakage_frame_grid_bitflip_exact-identity_paper.json").read_text())
    cmp_ = [((r["leak"], r["strength"], r["spread"]), r["loss"]) for r in ref["points"]]
    print(
        "pooled loss vs tableau_ref.py's exact subset (independent random numbers): "
        + ", ".join(f"{p} {table[p]['pooled'][0]:.3f}/{l:.3f}" for p, l in cmp_ if p in table)
    )

    print(
        "\nloss by point, sorted by the pooled loss (co-streak x1e-4: pooled / per slot, simulated)"
    )
    for p in sorted(allp, key=lambda q: table[q]["pooled"][0]):
        sim = sims[p]
        print(
            f"  {p!s:24s} pooled {table[p]['pooled'][0]:.3f}  slot {table[p]['slot'][0]:.3f}"
            f"   co_data {1e4 * sim['Z co_data']:.2f}/{1e4 * sim['Z co_data_slot']:.2f}"
            f"  co_gauge {1e4 * sim['Z co_gauge']:.2f}/{1e4 * sim['Z co_gauge_slot']:.2f}"
        )
    best = {}
    keysets = {"published grid": paper, "all points": allp}
    for v in FIT_VARIANTS:
        for n, keys in keysets.items():
            if not keys:
                continue
            p = argmin(table, v, keys)
            ranked = sorted(keys, key=lambda q: table[q][v][0])
            best[f"{v} {n}"] = p
            print(
                f"best ({v}, {n}): {p} loss {table[p][v][0]:.3f}; runner-up {ranked[1]} {table[ranked[1]][v][0]:.3f}  | residuals "
                + "  ".join(f"{k} {100 * x:+.0f}%" for k, x in table[p][v][1].items())
            )

    # Monte Carlo spread where several seeds exist
    reps = collections.defaultdict(list)
    for q in exact_grid_all():
        if q["shots"] == 8000:
            reps[(q["leak"], q["strength"], q["spread"])].append(q)
    mc = {}
    for p, qs in reps.items():
        if len(qs) > 1:
            lp = [E.loss(q["sim"], dev, "pooled")[0] for q in qs]
            ls = [E.loss(q["sim"], dev, "slot")[0] for q in qs]
            mc[str(p)] = {"seeds": [q["seed"] for q in qs], "pooled": lp, "slot": ls}
            print(
                f"Monte Carlo: {p} over seeds {[q['seed'] for q in qs]}: pooled {np.mean(lp):.3f} +- {np.std(lp, ddof=1):.3f}, "
                f"slot {np.mean(ls):.3f} +- {np.std(ls, ddof=1):.3f}"
            )

    # Monte Carlo replicates of the leading points: two more seeds each (the same simulator)
    life, read = E.hmm_inputs()
    lead = []  # the published grid's three best points under each statistic
    for v in FIT_VARIANTS:
        lead += sorted(paper, key=lambda q: table[q][v][0])[:3]
    lead = list(dict.fromkeys(lead))
    E.ensure(
        [
            ("det", j, b, 8, Point(*p, life, read), E.GRID_SHOTS, s)
            for p in lead
            for s in MC_SEEDS
            for j, b in CASES
        ],
        a.workers,
    )
    mc_rep = {}
    for p in lead:
        sims_p = [sims[p]] + [
            merged(
                case_stats(
                    {c: E.load(*c, 8, Point(*p, life, read), E.GRID_SHOTS, s) for c in CASES},
                    labels,
                )
            )
            for s in MC_SEEDS
        ]
        mc_rep[p] = {v: [E.loss(x, dev, v)[0] for x in sims_p] for v in FIT_VARIANTS}
        mc_rep[p]["sims"] = sims_p
    print(
        f"\nMonte Carlo replicates (seed 21 from the grid, seeds {MC_SEEDS} here), loss mean +- sd over the 3:"
    )
    for p in lead:
        print(
            f"  {p!s:24s} pooled {np.mean(mc_rep[p]['pooled']):.3f} +- {np.std(mc_rep[p]['pooled'], ddof=1):.3f}"
            f"   slot {np.mean(mc_rep[p]['slot']):.3f} +- {np.std(mc_rep[p]['slot'], ddof=1):.3f}"
            f"   ({', '.join(f'{x:.3f}' for x in mc_rep[p]['pooled'])} | {', '.join(f'{x:.3f}' for x in mc_rep[p]['slot'])})"
        )
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
        print(f"  {k:24s} " + ", ".join(f"{p} {n / a.boot:.0%}" for p, n in c.most_common(5)))
    for n in same:
        print(
            f"  {n}: the round-conditioned argmin equals the pooled one in {same[n] / a.boot:.0%} of replicates"
        )
    for v in FIT_VARIANTS:
        print(
            f"  {v}: loss at {BEST_PUBLISHED}, 95% bootstrap range {np.percentile(loss_best[v], 2.5):.3f}-{np.percentile(loss_best[v], 97.5):.3f}"
        )
    se = {
        k: float(np.std([d[k] for d in dev_reps], ddof=1) / abs(dev[k]))
        for k in [*E.FIT_KEYS, "Z co_data_slot", "Z co_gauge_slot"]
    }
    print(
        "  device relative standard errors: "
        + "  ".join(f"{k} {100 * v:.1f}%" for k, v in se.items())
    )

    out = {
        "device": dev,
        "device_rel_se": se,
        "grid_source": "leakage_grid_identity.json (seed 21, 8000 shots)",
        "points": [
            {
                "point": list(p),
                "loss_pooled": table[p]["pooled"][0],
                "loss_slot": table[p]["slot"][0],
                "rel_pooled": table[p]["pooled"][1],
                "rel_slot": table[p]["slot"][1],
                "co_data": sims[p]["Z co_data"],
                "co_data_slot": sims[p]["Z co_data_slot"],
                "co_gauge": sims[p]["Z co_gauge"],
                "co_gauge_slot": sims[p]["Z co_gauge_slot"],
            }
            for p in allp
        ],
        "best": {k: list(v) for k, v in best.items()},
        "monte_carlo": mc,
        "monte_carlo_leading": {str(p): {v: mc_rep[p][v] for v in FIT_VARIANTS} for p in lead},
        "bootstrap": {
            "B": a.boot,
            "wins": {k: {str(p): n for p, n in c.items()} for k, c in wins.items()},
            "same_argmin_share": {n: same[n] / a.boot for n in same},
            "loss_at_published_optimum_95": {
                v: np.percentile(loss_best[v], [2.5, 97.5]).tolist() for v in FIT_VARIANTS
            },
        },
    }
    (RESULTS / "leakage_roundcond.json").write_text(json.dumps(out, indent=1, default=float) + "\n")
    print("\nwrote", RESULTS / "leakage_roundcond.json")


if __name__ == "__main__":
    main()
