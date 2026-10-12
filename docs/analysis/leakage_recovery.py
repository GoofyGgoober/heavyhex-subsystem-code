"""Parameter recovery: simulate with known leakage parameters, run the fits, see what comes back.

Synthetic "devices" are exact simulations (leakage.ensure: exact_sim.py, a CZ with a leaked
operand does not act) of the four fit jobs' 8-round X- and Z-memory circuits with the
device's shot count (2500 per circuit) and seeds other than the grid's, at three known points:

  A  (leak 0.2%, partner X error 25%, transfer 5%)  the published optimum
  B  (0.3%, 25%, 0)
  C  (0.1%, 25%, 5%)

all with the paper's lifetime (35.4 us) and leaked readout (0.508), NREP datasets each.
Each dataset is then fitted exactly as the device was: its six statistics against the
exact grid (results/leakage_grid_identity.json, seed 21, 8000 shots per circuit; other
seeds, so grid and data share no random numbers), argmin of the loss, with the pooled and
with the round-conditioned co-streak statistic.

For truth A the 4- and 6-round Z-memory circuits are simulated as well, and the hidden
Markov fit of hmm.py (hmm_bootstrap.em_batch: the published procedure, and the
same run to convergence) is run on every flag and relay series of each dataset, to see
whether it returns the simulator's leaked-state lifetime and readout; the device's own
series of the same four jobs are fitted alongside, and the re-firing excess at lags 1-6
and the firing rate by round are compared (a posterior-predictive check of the fitted
leakage model). The simulator's leaked qubit returns with probability
1 - exp(-3.95 us / 35.4 us) at each of the two TICKs of a round, i.e. b = 0.200 per round
(7.9 us / b = 39.5 us), and reads 1 with probability 0.508.

Heavy the first time (the simulations are cached).

    python docs/analysis/leakage_recovery.py [--workers 2] [--nrep 4]

Writes results/leakage_recovery.json.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys

import numpy as np

import hmm_bootstrap as H
import leakage as E
from common import RESULTS, ROUND_US, WORKERS, jobs, memory_index, readout_series
from leakage import (
    CASES,
    FIT_JOBS,
    PAPER_GRID,
    Point,
    argmin,
    case_stats,
    device_cases,
    exact_grid,
    fit_table,
    merged,
)

TRUTHS = {"A": (0.002, 0.25, 0.05), "B": (0.003, 0.25, 0.0), "C": (0.001, 0.25, 0.05)}
SHOTS = 2500
SEED0 = 101


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--nrep", type=int, default=4)
    ap.add_argument("--sim-only", action="store_true")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    life, read = E.hmm_inputs()
    seeds = [SEED0 + i for i in range(a.nrep)]
    tasks = []
    for t, p in TRUTHS.items():
        pt = Point(*p, life, read)
        for s in seeds:
            tasks += [("det", j, b, 8, pt, SHOTS, s) for j, b in CASES]
            if t == "A":
                tasks += [("det", j, "Z", r, pt, SHOTS, s) for j in FIT_JOBS for r in (4, 6)]
    E.ensure(tasks, a.workers)
    if a.sim_only:
        return
    # the grid the device fit uses: the exact grid (seed 21, 8000 shots)
    labels = {c: lab for c, (_, lab) in device_cases().items()}
    sims = exact_grid()
    keysets = {"published grid": [p for p in PAPER_GRID if p in sims], "all points": list(sims)}
    print(
        f"grid: {len(sims)} points from leakage_grid_identity.json ({len(keysets['published grid'])} published-grid points)"
    )

    res = {"grid_fit": {}, "hmm": {}}
    print(f"grid fit of synthetic devices ({SHOTS} shots per circuit, {a.nrep} datasets per truth)")
    for t, p in TRUTHS.items():
        pt = Point(*p, life, read)
        rows = []
        for s in seeds:
            syn = merged(case_stats({c: E.load(*c, 8, pt, SHOTS, s) for c in CASES}, labels))
            tb = fit_table(sims, syn)
            row = {"seed": s}
            for v in ("pooled", "slot"):
                for n, keys in keysets.items():
                    row[f"{v} {n}"] = list(argmin(tb, v, keys))
                row[f"{v} loss_at_truth"] = tb[p][v][0]
                row[f"{v} loss_best"] = tb[tuple(row[f"{v} all points"])][v][0]
                ranked = sorted(keysets["all points"], key=lambda q: tb[q][v][0])
                row[f"{v} rank_of_truth"] = ranked.index(p) + 1
            rows.append(row)
            print(
                f"  truth {t} {p} seed {s}: pooled -> {tuple(row['pooled all points'])} "
                f"(truth ranked {row['pooled rank_of_truth']}, loss {row['pooled loss_at_truth']:.3f} vs best {row['pooled loss_best']:.3f}); "
                f"slot -> {tuple(row['slot all points'])} (truth ranked {row['slot rank_of_truth']})",
                flush=True,
            )
        for v in ("pooled", "slot"):
            for n in keysets:
                c = collections.Counter(tuple(r[f"{v} {n}"]) for r in rows)
                print(
                    f"    {v:6s} {n:15s}: "
                    + ", ".join(f"{q} {k}/{len(rows)}" for q, k in c.most_common())
                )
        res["grid_fit"][t] = {"truth": list(p), "rows": rows}

    # hidden Markov fit on synthetic series (truth A), and the same on the device's fit jobs
    pt = Point(*TRUTHS["A"], life, read)
    js = {j.name: j for j in jobs() if j.name in FIT_JOBS}
    b_true = 1 - np.exp(-ROUND_US / life)

    def fits(codes):
        counts = {T: H.counts_of(codes[T], T) for T in H.LENGTHS}
        pub, _, _, capped = H.em_batch(counts, H.paper_init(counts))
        conv, _, _, capc = H.em_batch(
            counts, H.paper_init(counts), iters=H.CONV_ITERS, tol=H.CONV_TOL
        )
        m = {
            "published": H.summary(pub),
            "converged": H.summary(H.relabel(conv)),
            "capped_300": int(capped.sum()),
            "capped_conv": int(capc.sum()),
            "n_series": len(codes[8]),
        }
        x8 = H.decode(codes[8], 8)  # series x shots x 8
        m["refire"] = [float(np.mean([H.refire(x, k) for x in x8])) for k in range(1, 7)]
        m["rate_by_round"] = x8.mean((0, 1)).tolist()
        return m

    syn = []
    for s in seeds:
        codes = {T: [] for T in H.LENGTHS}
        for name in FIT_JOBS:
            per = collections.defaultdict(dict)
            for T in H.LENGTHS:
                _, _, lab = js[name].model(memory_index("Z", T))
                det = E.load(name, "Z", T, pt, SHOTS, s)
                for key, cols in readout_series(lab).items():
                    per[key][T] = (det[:, cols].astype(np.int64) << np.arange(T)).sum(1)
            for _key, d in per.items():
                for T in H.LENGTHS:
                    codes[T].append(d[T])
        m = fits({T: np.stack(v) for T, v in codes.items()})
        m["seed"] = s
        syn.append(m)
        for how in ("published", "converged"):
            q = m[how]
            print(
                f"  HMM ({how} procedure) on synthetic series, truth A seed {s} ({m['n_series']} series): median b {q['b']:.4f} "
                f"(simulator {b_true:.4f}); 7.9 us x median 1/b = {q['life_us']:.1f} us (input {life:.1f} us, simulator's mean "
                f"stay {ROUND_US / b_true:.1f} us); e_P {q['e_P']:.3f} (input {read:.3f}); a {q['a']:.4f}; e_N {q['e_N']:.3f}",
                flush=True,
            )
    dcodes, dmeta = H.load_series()
    keep = np.array([m["job"] in FIT_JOBS for m in dmeta])
    dev = fits({T: dcodes[T][keep] for T in H.LENGTHS})
    for how in ("published", "converged"):
        q = dev[how]
        print(
            f"  device, the same {dev['n_series']} series of the fit jobs ({how}): median b {q['b']:.4f}, 7.9 us x median 1/b = "
            f"{q['life_us']:.1f} us, e_P {q['e_P']:.3f}, a {q['a']:.4f}, e_N {q['e_N']:.3f}"
        )
    mean = {
        how: {k: float(np.mean([m[how][k] for m in syn])) for k in H.KEYS}
        for how in ("published", "converged")
    }
    for how in ("published", "converged"):
        print(
            f"  synthetic mean over datasets ({how}): b {mean[how]['b']:.4f}, lifetime {mean[how]['life_us']:.1f} us, "
            f"e_P {mean[how]['e_P']:.3f}, a {mean[how]['a']:.4f}"
        )
    ref_s = np.mean([m["refire"] for m in syn], axis=0)
    print("  re-firing excess at lag 1..6 (points), 8-round Z series of the fit jobs:")
    print("    device    " + " ".join(f"{100 * v:6.2f}" for v in dev["refire"]))
    print(
        "    synthetic "
        + " ".join(f"{100 * v:6.2f}" for v in ref_s)
        + "   (truth A, mean of the datasets)"
    )
    print("  firing rate by round (%):")
    print("    device    " + " ".join(f"{100 * v:5.2f}" for v in dev["rate_by_round"]))
    print(
        "    synthetic "
        + " ".join(f"{100 * v:5.2f}" for v in np.mean([m["rate_by_round"] for m in syn], axis=0))
    )
    res["hmm"] = {
        "truth": {
            "b_per_round": float(b_true),
            "life_input_us": life,
            "mean_stay_us": float(ROUND_US / b_true),
            "read_leaked": read,
        },
        "datasets": syn,
        "mean": mean,
        "device_fit_jobs": dev,
    }
    (RESULTS / "leakage_recovery.json").write_text(json.dumps(res, indent=1, default=float) + "\n")
    print("wrote", RESULTS / "leakage_recovery.json")


if __name__ == "__main__":
    main()
