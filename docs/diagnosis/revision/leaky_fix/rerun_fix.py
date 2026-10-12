"""rerun.py's fit / fit-bitflip variant, with the fixed simulator, and Table IV's numbers.

Simulates every requested job's memory circuits (both bases, every round count) with
leaky_fix/leaky.py at the given leakage parameters, decodes exactly as rerun.py does
(R._decode with the calibration model's priors, seed = rounds, 10,000 shots), fits eps
with fit_per_round, and takes the device ("chip") and calibration-model ("model") entries
from the original rerun output (leakage off, so unaffected by the fix).

    python rerun_fix.py TAG --kind bitflip --mode dephase --leak 0.002 --strength 0.25 --spread 0.05
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DIAG = HERE.parents[1]
sys.path[:0] = [str(HERE), str(DIAG)]
ORIG = DIAG / "results/rerun_model-fit-bitflip_cz1.5_d1_p0.5.json"
ORIG_DEP = DIAG / "results/rerun_model-fit_cz1.5_d1_p0.5.json"


def _job(args):
    name, kind, mode, leak, strength, spread, shots = args
    import leaky
    assert Path(leaky.__file__).resolve().parent == HERE
    import rerun
    from common import ROUND_US, jobs, memory_index
    from heavyhex.experiment import run as R
    from heavyhex.experiment.circuits import ROUNDS
    from heavyhex.simulation.noisy import fit_per_round
    job = next(j for j in jobs() if j.name == name)
    _, _, life, read = rerun.leakage_params()
    q = rerun.PROPS[job.day]["qubits"]
    lk = leaky.Leakage(
        leak=rerun._AllQubits(defaultdict(lambda: leak)), lifetime_us=life, half_round_us=ROUND_US / 2,
        read_leaked=read, partner_phase=strength, partner_kind=kind, transfer=spread,
        reset_error={int(k): v["init_error"] for k, v in q.items() if v.get("init_error") is not None},
        readout={int(k): (v["prob_meas1_prep0"], v["prob_meas0_prep1"]) for k, v in q.items()
                 if v.get("prob_meas1_prep0") is not None},
        skipped_cz="dephase" if mode.startswith("exact-") else mode)
    errors = {b: defaultdict(list) for b in "XZ"}
    for basis in "XZ":
        for rounds in ROUNDS:
            model, records, labels = job.model(memory_index(basis, rounds))
            if mode.startswith("exact-"):  # exact stabilizer simulation, tableau_ref.py
                import tableau_ref
                det, obs = tableau_ref.simulate(model, shots, lk, mode.removeprefix("exact-"), seed=rounds)
            else:
                det, obs = leaky.simulate(model, shots, lk, seed=rounds)
            failed = R._decode(model, det) != obs
            patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
            for k, patch in enumerate(patches):
                errors[basis][patch].append(failed[:, k].mean())
    fits = {b: {p: fit_per_round(ROUNDS, e, shots) for p, e in errors[b].items()} for b in "XZ"}
    return name, {b: {p: [f.per_round, f.uncertainty] for p, f in fits[b].items()} for b in "XZ"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--kind", default="bitflip")
    ap.add_argument("--mode", default="dephase")
    ap.add_argument("--leak", type=float, default=0.002)
    ap.add_argument("--strength", type=float, default=0.25)
    ap.add_argument("--spread", type=float, default=0.05)
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--jobs", nargs="*")
    a = ap.parse_args()
    orig = json.loads(ORIG.read_text())
    names = a.jobs or list(orig["jobs"])
    with mp.Pool(min(12, len(names))) as pool:
        res = dict(pool.map(_job, [(n, a.kind, a.mode, a.leak, a.strength, a.spread, a.shots) for n in names], chunksize=1))
    out = {"params": vars(a), "jobs": {}}
    for n in names:
        out["jobs"][n] = {"model": orig["jobs"][n]["model"], "chip": orig["jobs"][n]["chip"], "leak": res[n]}
    (HERE / f"rerun_{a.tag}.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", HERE / f"rerun_{a.tag}.json")


if __name__ == "__main__":
    main()
