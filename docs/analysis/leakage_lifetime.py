"""What the hidden Markov fit returns as a function of the simulator's leaked-state lifetime.

Follow-up to leakage_recovery.py, which found that the hidden Markov fit, run on flag and relay
series simulated from the fitted leakage model (lifetime 35.4 us, leaked readout 0.508),
returns b ~ 0.12 per round (7.9 us / b ~ 64 us) and e_P ~ 0.44, not the simulator's
b = 0.20 and 0.508, while the device gives b = 0.22. Here the same model (leak 0.2%,
partner X error 25%, transfer 5%, leaked readout 0.508) is simulated exactly at shorter
lifetimes, on the 4-, 6- and 8-round Z-memory circuits of the four fit jobs (2500 shots,
seed 101; leakage.ensure), and fitted with the published procedure, to find the simulator
lifetime whose fitted b matches the device's. So the paper's 35 us is an effective
persistence, not the leaked state's lifetime. Heavy the first time (the simulations are cached).

    python docs/analysis/leakage_lifetime.py [--workers 2]

Writes results/leakage_lifetime.json.
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
from leakage import FIT_JOBS, Point

LIVES = (10.0, 17.7, 25.0)  # us; 35.4 us (the paper's) is leakage_recovery.py's truth A
SEED, SHOTS = 101, 2500


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    life0, read = E.hmm_inputs()
    lives = [*LIVES, life0]
    pts = {L: Point(0.002, 0.25, 0.05, L, read) for L in lives}
    E.ensure(
        [
            ("det", j, "Z", T, pt, SHOTS, SEED)
            for pt in pts.values()
            for j in FIT_JOBS
            for T in H.LENGTHS
        ],
        a.workers,
    )
    js = {j.name: j for j in jobs() if j.name in FIT_JOBS}
    labels = {(n, T): js[n].model(memory_index("Z", T))[2] for n in FIT_JOBS for T in H.LENGTHS}
    out = {}
    for L, pt in pts.items():
        codes = {T: [] for T in H.LENGTHS}
        for n in FIT_JOBS:
            per = collections.defaultdict(dict)
            for T in H.LENGTHS:
                det = E.load(n, "Z", T, pt, SHOTS, SEED)
                for key, cols in readout_series(labels[(n, T)]).items():
                    per[key][T] = (det[:, cols].astype(np.int64) << np.arange(T)).sum(1)
            for _key, d in per.items():
                for T in H.LENGTHS:
                    codes[T].append(d[T])
        counts = {T: H.counts_of(np.stack(codes[T]), T) for T in H.LENGTHS}
        p_, _, _, _ = H.em_batch(counts, H.paper_init(counts))
        m = H.summary(p_)
        x8 = H.decode(np.stack(codes[8]), 8)
        m["refire"] = [float(np.mean([H.refire(x, k) for x in x8])) for k in range(1, 7)]
        m["sim_b"] = float(1 - np.exp(-ROUND_US / L))
        out[f"{L:.2f}"] = m
        print(
            f"simulator lifetime {L:5.1f} us (b {m['sim_b']:.3f} per round): fitted median b {m['b']:.4f} "
            f"(7.9 us / b = {m['life_us']:.1f} us), e_P {m['e_P']:.3f}, a {m['a']:.4f}; re-firing excess lag 1..6 "
            + " ".join(f"{100 * v:.2f}" for v in m["refire"]),
            flush=True,
        )
    (RESULTS / "leakage_lifetime.json").write_text(json.dumps(out, indent=1) + "\n")
    print(
        "device (fit jobs): median b 0.2245, e_P 0.512, a 0.0258; re-firing excess 14.87 10.74 7.72 5.75 4.09 3.03"
    )


if __name__ == "__main__":
    main()
