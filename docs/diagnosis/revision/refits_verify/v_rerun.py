"""Verifier for the rerun-table refit.

(a) Fit the main analysis's regenerated per-round failure counts (../refits/rerun_per_round.json)
    with MY weighted-log fit and compare with the stored eps in
    docs/diagnosis/results/rerun_model-fit-bitflip_cz1.5_d1_p0.5.json.
(b) Independently regenerate the counts of a few jobs with rerun.py's own variant
    definitions and seeds (seed = rounds) and compare count by count.
(c) Refit everything with my binomial MLE and form the table means (eps: mean over
    the 10 reps; Lambda: mean of each rep's eps3/eps5).
Usage: python v_rerun.py [job names to regenerate ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "docs/diagnosis"))
from v_fit import ROUNDS, mle, wls  # noqa: E402

theirs = json.loads((REPO / "docs/diagnosis/revision/refits/rerun_per_round.json").read_text())
stored = json.loads((REPO / "docs/diagnosis/results/rerun_model-fit-bitflip_cz1.5_d1_p0.5.json").read_text())
N = theirs["shots"]
assert N == stored["shots"] == 10000 and theirs["rounds"] == [1, 2, 3, 4, 6, 8]

# (a)
worst = 0.0
for name, j in theirs["jobs"].items():
    for v, by in j["failures"].items():
        for b, pp in by.items():
            for p, k in pp.items():
                e = wls(ROUNDS, np.array(k) / N, N)["eps"]
                worst = max(worst, abs(e - stored["jobs"][name][v][b][p][0]))
print(f"(a) my WLS on their regenerated counts vs stored eps: max |diff| = {worst:.2e}")
assert set(theirs["jobs"]) == set(stored["jobs"])

# (b)
regen = sys.argv[1:]
if regen:
    import leaky  # noqa: E402
    import rerun as RR  # noqa: E402
    from common import jobs, memory_index  # noqa: E402
    from heavyhex.experiment import run as R  # noqa: E402

    RR.KNOBS.update(stored["knobs"])
    params = RR.leakage_params()
    for job in jobs():
        if job.name not in regen:
            continue
        for v in ("model", "fit-bitflip"):
            for basis in "XZ":
                for i, rounds in enumerate(ROUNDS.astype(int)):
                    model, records, labels = job.model(memory_index(basis, int(rounds)))
                    det, obs = leaky.simulate(model, N, RR.variant(job, v, *params, stored["cz_scale"]), seed=int(rounds))
                    failed = R._decode(model, det) != obs
                    patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
                    for k, patch in enumerate(patches):
                        mine = int(failed[:, k].sum())
                        th = theirs["jobs"][job.name]["failures"][v][basis][patch][i]
                        print(f"(b) {job.name} {v} {basis} d{patch} n={rounds}: mine {mine} theirs {th} {'OK' if mine == th else 'MISMATCH'}", flush=True)

# (c)
rows = defaultdict(lambda: defaultdict(dict))  # (day, rep) -> src -> key -> (wls, mle)
for name, j in theirs["jobs"].items():
    day, rep = name[:10], int(name[12])
    for v, by in j["failures"].items():
        for b, pp in by.items():
            for p, k in pp.items():
                k = np.array(k, float)
                rows[(day, rep)][v][f"{b}{p}"] = (wls(ROUNDS, k / N, N)["eps"], mle(k, N)["eps"])
assert len(rows) == 10
print("\n(c) table means, MLE (WLS):")
for v in ("model", "fit-bitflip"):
    line = f"  {v:12s}"
    for key in ("X3", "X5", "Z3", "Z5"):
        w = np.mean([rows[r][v][key][0] for r in rows]) * 100
        m = np.mean([rows[r][v][key][1] for r in rows]) * 100
        line += f" {key} {m:.2f} ({w:.2f})"
    for b in "XZ":
        w = np.mean([rows[r][v][f"{b}3"][0] / rows[r][v][f"{b}5"][0] for r in rows])
        m = np.mean([rows[r][v][f"{b}3"][1] / rows[r][v][f"{b}5"][1] for r in rows])
        line += f" L{b} {m:.4f} ({w:.4f})"
    print(line)
