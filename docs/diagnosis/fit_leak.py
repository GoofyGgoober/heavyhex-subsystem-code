"""Fit the two leakage inputs the flags and relays cannot measure, on detector rates only.

data_scale: the leak rate per CZ of data qubits and Z ancillas, as a share of the median
measured on the flags and relays. partner_phase: how often a leaked qubit's CZ dephases
its partner. Both are set by matching the chip's mean detection rate per round in three
groups (X-memory stabilizers, Z-memory stabilizers, Z-memory flags and relays) in the
8-round runs. The logical errors and Λ are left out of the fit, so rerun.py can test
the fitted model on them.

    python docs/diagnosis/fit_leak.py
"""

from __future__ import annotations

import itertools
import json

import numpy as np

from common import REPO, jobs, memory_index
import leaky
import rerun

FIT_JOBS = ("2026-10-08-r1", "2026-10-08-r2-d5", "2026-10-08-r2-d3", "2026-10-06-r1")
SCALES = (0.0, 0.25, 0.5, 1.0)
PHASES = (0.0, 0.1, 0.25, 0.5)
SHOTS = 4000


def groups(labels):
    g = {"stab": [], "flag": []}
    for col, (patch, rnd, name) in enumerate(labels):
        if " " in name:
            g["flag"].append(col)
        elif 1 <= rnd <= 7:
            g["stab"].append(col)
    return g


def main() -> None:
    params = rerun.leakage_params()
    js = [j for j in jobs() if j.name in FIT_JOBS]
    targets, sims = [], {}
    for j in js:
        for basis in "XZ":
            idx = memory_index(basis, 8)
            model, records, labels = j.model(idx)
            chip, _ = j.chip_detectors(idx, model, records)
            for g, cols in groups(labels).items():
                if cols:
                    targets.append((j, basis, idx, g, cols, chip[:, cols].mean()))
    results = []
    for scale, phase in itertools.product(SCALES, PHASES):
        rerun.KNOBS.update(data_scale=scale, partner_phase=phase)
        cache, sq = {}, []
        rows = []
        for j, basis, idx, g, cols, obs in targets:
            if (j.name, basis) not in cache:
                model, _, _ = j.model(idx)
                det, _ = leaky.simulate(model, SHOTS, rerun.variant(j, "leak+ro", *params, 1.0), seed=11)
                cache[(j.name, basis)] = det
            sim = cache[(j.name, basis)][:, cols].mean()
            sq.append((sim - obs) ** 2); rows.append((j.name, basis, g, obs, sim))
        rms = float(np.sqrt(np.mean(sq)))
        results.append({"data_scale": scale, "partner_phase": phase, "rms": rms, "rows": rows})
        by = {}
        for _, basis, g, obs, sim in rows:
            by.setdefault(f"{basis} {g}", []).append(sim - obs)
        print(f"data_scale {scale:<4g} partner_phase {phase:<4g} rms {rms:.4f}  " +
              "  ".join(f"{k}: {np.mean(v):+.3f}" for k, v in sorted(by.items())), flush=True)
    best = min(results, key=lambda r: r["rms"])
    print(f"\nbest: data_scale {best['data_scale']}, partner_phase {best['partner_phase']} (rms {best['rms']:.4f})")
    (REPO / "docs/diagnosis/results/fit_leak.json").write_text(json.dumps(results, indent=1, default=str) + "\n")


if __name__ == "__main__":
    main()
