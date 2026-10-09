"""Where would these circuits reach Λ = 1? Every error rate at Fez's median, scaled together.

A threshold proper needs d >= 7, which these circuits don't go to. What d = 3 and 5 can
give is the scale s at which Λ = ε₃/ε₅ crosses 1 when every CZ, readout, reset,
single-qubit error and idle time on a uniform chip at the 8 October medians is multiplied
by s (decoupled waits, f = 1, the run's median). s = 1 is today's median chip.

    python docs/diagnosis/uniform_scale.py   # writes docs/diagnosis/results/uniform_scale.json
"""

from __future__ import annotations

import json

import numpy as np

from common import REPO, jobs
from heavyhex.simulation.noisy import decay, typical

SCALES = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0)


def main() -> None:
    cal = typical(jobs()[-1].calibration)
    out = {"calibration": jobs()[-1].name, "scales": SCALES, "f": 1.0}
    for basis in "XZ":
        lam = []
        for s in SCALES:
            e = {d: decay(d, basis, cal, shots=20000, seed=5, scale=s, decoupling=True, decoupled_dephasing=1.0).per_round
                 for d in (3, 5)}
            lam.append(e[3] / e[5])
            print(f"{basis} memory, scale {s:.2f}: eps3 {100*e[3]:.2f}%  eps5 {100*e[5]:.2f}%  Λ {e[3]/e[5]:.2f}", flush=True)
        lam = np.array(lam)
        cross = float(np.interp(1.0, lam[::-1], np.array(SCALES)[::-1])) if lam.min() < 1 < lam.max() else None
        out[basis] = {"lambda": lam.tolist(), "crosses_at": cross}
        print(f"{basis} memory: Λ = 1 at scale {cross}", flush=True)
    (REPO / "docs/diagnosis/results/uniform_scale.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
