"""Regenerate the per-round P(n) behind the paper's Table "rerun" (calibration model and the
fitted leakage model, fit-bitflip), which docs/diagnosis/results/rerun_*.json store only as
fitted eps. Uses docs/diagnosis/rerun.py's own variants, shot count (10,000) and seeds
(seed = round count), so the samples are the same ones; the prespecified fit of these P(n)
must reproduce the stored eps exactly, which is checked. Writes rerun_per_round.json here.

    python docs/diagnosis/revision/refits/rerun_per_round.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO / "docs/diagnosis"))

import leaky  # noqa: E402
import rerun as RR  # noqa: E402
from common import jobs, memory_index  # noqa: E402

from heavyhex.experiment import run as R  # noqa: E402
from heavyhex.experiment.circuits import ROUNDS  # noqa: E402
from heavyhex.simulation.noisy import fit_per_round  # noqa: E402

SHOTS = 10_000
CZ_SCALE = 1.5
VARIANTS = ("model", "fit-bitflip")
STORED = REPO / "docs/diagnosis/results/rerun_model-fit-bitflip_cz1.5_d1_p0.5.json"


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
                    model, records, labels = job.model(memory_index(basis, rounds))
                    det, obs = leaky.simulate(model, SHOTS, RR.variant(job, v, *params, CZ_SCALE), seed=rounds)
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
        print(f"{job.name}: {time.time() - t0:.0f} s, worst |eps - stored| so far {worst:.2e}", flush=True)
    out["max_abs_difference_from_stored_eps"] = worst
    (HERE / "rerun_per_round.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
