"""The failure counts at every round count behind leakage_frame_rerun_bitflip.json, which
stores only the fitted eps: the calibration model and the fitted leakage model (fit-bitflip),
rerun with leakage_frame_rerun.py's own variants, shots (10,000) and seeds (seed = round
count), so the samples are the same ones. The prespecified fit of these counts must give the
stored eps back exactly, which is checked. leakage_fit.py refits them by binomial likelihood.

Heavy (two variants x 12 circuits x 15 jobs, 10,000 shots each).

    python docs/analysis/leakage_frame_counts.py     # writes results/leakage_frame_counts.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict

from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import ROUNDS
from heavyhex.simulation.noisy import fit_per_round

import frame_sim
import leakage_frame_rerun as RR
from common import RESULTS, jobs, memory_index

SHOTS = 10_000
CZ_SCALE = 1.5
VARIANTS = ("model", "fit-bitflip")
STORED = RESULTS / "leakage_frame_rerun_bitflip.json"


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    stored = json.loads(STORED.read_text())
    assert stored["shots"] == SHOTS and stored["cz_scale"] == CZ_SCALE
    RR.KNOBS.update(stored["knobs"])
    params = RR.leakage_params()
    out = {"shots": SHOTS, "rounds": list(ROUNDS), "variants": list(VARIANTS), "jobs": {}}
    worst = 0.0
    for job in jobs():
        t0 = time.time()
        found = {}
        for v in VARIANTS:
            errors = {b: defaultdict(list) for b in "XZ"}
            for basis in "XZ":
                for rounds in ROUNDS:
                    model, _records, labels = job.model(memory_index(basis, rounds))
                    det, obs = frame_sim.simulate(
                        model, SHOTS, RR.variant(job, v, *params, CZ_SCALE), seed=rounds
                    )
                    failed = R._decode(model, det) != obs
                    patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
                    for k, patch in enumerate(patches):
                        errors[basis][patch].append(int(failed[:, k].sum()))
            found[v] = {b: dict(e) for b, e in errors.items()}
            for b in "XZ":
                for p, k in errors[b].items():
                    f = fit_per_round(ROUNDS, [x / SHOTS for x in k], SHOTS)
                    ref = stored["jobs"][job.name][v][b][p][0]
                    worst = max(worst, abs(f.per_round - ref))
        out["jobs"][job.name] = {"day": job.day, "rep": job.rep, "failures": found}
        print(
            f"{job.name}: {time.time() - t0:.0f} s, worst |eps - stored| so far {worst:.2e}",
            flush=True,
        )
    out["max_abs_difference_from_stored_eps"] = worst
    (RESULTS / "leakage_frame_counts.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
