"""How well the device's six detection statistics pin down the exact grid's optimum.

Resamples the device's shots within each of the fit's eight circuits (B replicates),
recomputes the six statistics (pooled and per-slot co-streak), and takes the argmin of the
loss over the points of a grid (results/leakage_grid_<CHANNEL>.json, seed 21, 8000 shots;
simulated statistics held fixed). Reports how often each point wins and the device-noise
floor (the loss of a replicate against the full-data device statistics). The Monte Carlo
noise of the simulation is in leakage_fit.py (seed replicates).

    python docs/analysis/leakage_grid_bootstrap.py identity [B] [--workers 2]

Writes results/leakage_grid_bootstrap_<CHANNEL>.json.
"""

from __future__ import annotations

import os

# one BLAS thread per process (OpenBLAS otherwise commits about 0.5 GB per process)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import collections
import json
import multiprocessing as mp
import sys

import numpy as np

from common import RESULTS, WORKERS, memory_index
from leakage_grid import POOLED, SLOT, case_stats, fit_cases, loss, merge

_C: list = []


def _init():
    for j, basis, _model, labels, *_ in fit_cases():
        chip, _ = j.chip_detectors(memory_index(basis, 8))
        _C.append((basis, labels, chip))


def _rep(seed):
    rng = np.random.default_rng(seed)
    return merge(case_stats(c[rng.integers(0, len(c), len(c))], b, lab) for b, lab, c in _C)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("channel")
    ap.add_argument("B", type=int, nargs="?", default=200)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    channel, B = a.channel, a.B
    g = json.loads((RESULTS / f"leakage_grid_{channel}.json").read_text())
    pts = [p for p in g["points"] if p["seed"] == 21 and p["shots"] == 8000]
    dev = g["device"]
    with mp.Pool(a.workers, initializer=_init) as pool:
        reps = pool.map(_rep, range(1000, 1000 + B), chunksize=4)
    out = {"channel": channel, "B": B, "n_points": len(pts)}
    for mode, keys, lk in (("pooled", POOLED, "loss"), ("per_slot", SLOT, "loss_slot")):
        wins = collections.Counter()
        floor = []
        for r in reps:
            best = min(pts, key=lambda p: loss(p["sim"], r, keys)[0])
            wins[(best["leak"], best["strength"], best["spread"])] += 1
            floor.append(loss(r, dev, keys)[0])
        sd = {k: float(np.std([r[k] for r in reps]) / abs(dev[k])) for k in keys}
        out[mode] = {
            "argmin_frequency": [[list(k), n / B] for k, n in wins.most_common()],
            "noise_floor_median": float(np.median(floor)),
            "noise_floor_p90": float(np.percentile(floor, 90)),
            "device_relative_sd": sd,
            "full_data_best": min(
                ([p["leak"], p["strength"], p["spread"], p[lk]] for p in pts), key=lambda x: x[3]
            ),
        }
        print(
            f"{channel} {mode}: device relative SD "
            + "  ".join(f"{k} {100 * v:.1f}%" for k, v in sd.items())
        )
        print(
            f"   noise floor of the loss: median {np.median(floor):.3f}, 90th pct {np.percentile(floor, 90):.3f}"
        )
        print(
            "   argmin frequency: "
            + ", ".join(f"{k}: {100 * n / B:.0f}%" for k, n in wins.most_common(6))
        )
    (RESULTS / f"leakage_grid_bootstrap_{channel}.json").write_text(
        json.dumps(out, indent=1) + "\n"
    )


if __name__ == "__main__":
    main()
