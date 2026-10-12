"""How well the device's six detection statistics pin down the exact-model grid optimum.

Resamples the device's shots within each of the fit's eight circuits (B replicates),
recomputes the six statistics (pooled and per-slot co-streak), and takes the argmin of the
loss over the points of a grid file (grid_<channel>.json, seed 21, 8000 shots; simulated
statistics held fixed). Reports how often each point wins and the device-noise floor (the
loss of a replicate against the full-data device statistics). Monte Carlo noise of the
simulation is reported separately (seed replicates in summarize_exact.py).

    python grid_bootstrap_exact.py CHANNEL [B]       (CHANNEL: identity | dephase_frame | pi ...)

Writes grid_bootstrap_<CHANNEL>.json next to this file.
"""

from __future__ import annotations

import os

# one BLAS thread per process: OpenBLAS otherwise commits ~0.5 GB per process on Windows,
# which the multiprocessing pools below cannot afford
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import collections
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIAG = HERE.parents[1]
sys.path[:0] = [str(HERE), str(DIAG)]

from grid_exact import POOLED, SLOT, case_stats, fit_cases, loss, merge  # noqa: E402

_C: list = []


def _init():
    from common import memory_index
    for j, basis, model, labels, *_ in fit_cases():
        chip, _ = j.chip_detectors(memory_index(basis, 8))
        _C.append((basis, labels, chip))


def _rep(seed):
    rng = np.random.default_rng(seed)
    return merge(case_stats(c[rng.integers(0, len(c), len(c))], b, lab) for b, lab, c in _C)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    channel = sys.argv[1]
    B = int(sys.argv[2]) if len(sys.argv) > 2 else 200
    g = json.loads((HERE / f"grid_{channel}.json").read_text())
    pts = [p for p in g["points"] if p["seed"] == 21 and p["shots"] == 8000]
    dev = g["device"]
    with mp.Pool(int(os.environ.get("NPROC", "4")), initializer=_init) as pool:
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
        out[mode] = {"argmin_frequency": [[list(k), n / B] for k, n in wins.most_common()],
                     "noise_floor_median": float(np.median(floor)), "noise_floor_p90": float(np.percentile(floor, 90)),
                     "device_relative_sd": sd,
                     "full_data_best": min(([p["leak"], p["strength"], p["spread"], p[lk]] for p in pts), key=lambda x: x[3])}
        print(f"{channel} {mode}: device relative SD " + "  ".join(f"{k} {100 * v:.1f}%" for k, v in sd.items()))
        print(f"   noise floor of the loss: median {np.median(floor):.3f}, 90th pct {np.percentile(floor, 90):.3f}")
        print("   argmin frequency: " + ", ".join(f"{k}: {100 * n / B:.0f}%" for k, n in wins.most_common(6)))
    (HERE / f"grid_bootstrap_{channel}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
