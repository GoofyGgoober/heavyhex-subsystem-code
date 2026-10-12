"""Table IV's leakage column with an exact simulation of the leakage channel.

Simulates every job's memory circuits (both bases, every round count) at the given leakage
parameters with exact_sim.py, decodes them exactly as rerun.py does (R._decode, the
calibration model's priors; seed = round count; 10,000 shots per circuit), and stores the
logical-failure counts at every round count, so that eps can be fitted both with the
prespecified weighted-log fit (heavyhex.simulation.noisy.fit_per_round) and by binomial
likelihood (refits/fitlib.py). Leakage settings as in rerun.py's fit-bitflip variant: every
qubit leaks with the same probability per CZ; partner X error and transfer as given; the
hidden Markov lifetime and leaked readout; each day's IBM reset errors and P(1|0), P(0|1).

    python rerun_exact.py TAG --channel identity --leak 0.002 --strength 0.25 --spread 0.05
           [--shots 10000] [--procs 13] [--jobs ...]

Writes rerun_<TAG>.json next to this file.
"""

from __future__ import annotations

import os

# one BLAS thread per process: OpenBLAS otherwise commits ~0.5 GB per process on Windows,
# which the multiprocessing pools below cannot afford
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path


HERE = Path(__file__).resolve().parent
DIAG = HERE.parents[1]
sys.path[:0] = [str(HERE), str(DIAG), str(DIAG / "revision/refits")]


def _task(args):
    """One job and basis, every round count (seed = round count, as rerun.py)."""
    folder, basis, channel, leak, strength, spread, shots, hmm, reset_error, readout = args
    import exact_sim as E
    from common import Job, memory_index
    from heavyhex.experiment import run as R
    from heavyhex.experiment.circuits import ROUNDS
    job = Job(Path(folder))
    lk = E.Leakage(leak=E.Uniform(leak), lifetime_us=hmm[0], half_round_us=hmm[2], read_leaked=hmm[1],
                   partner_phase=strength, partner_kind="bitflip", transfer=spread,
                   reset_error=reset_error, readout=readout)
    out = {}
    for rounds in ROUNDS:
        model, records, labels = job.model(memory_index(basis, rounds))
        det, obs = E.simulate(model, shots, lk, channel, seed=rounds)
        failed = R._decode(model, det) != obs
        patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
        out[rounds] = {p: int(failed[:, k].sum()) for k, p in enumerate(patches)}
    return job.name, basis, out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--channel", choices=["identity", "dephase", "pi"], default="identity")
    ap.add_argument("--leak", type=float, default=0.002)
    ap.add_argument("--strength", type=float, default=0.25)
    ap.add_argument("--spread", type=float, default=0.05)
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--procs", type=int, default=13)
    ap.add_argument("--jobs", nargs="*")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    import fitlib as F
    from common import jobs
    from heavyhex.experiment.circuits import ROUNDS
    from heavyhex.simulation.noisy import fit_per_round
    import rerun
    from grid_exact import hmm_params
    hmm = hmm_params()
    all_jobs = [j for j in jobs() if not a.jobs or j.name in a.jobs]
    names = [j.name for j in all_jobs]
    tasks = []
    for j in all_jobs:
        q = rerun.PROPS[j.day]["qubits"]
        re = {int(k): v["init_error"] for k, v in q.items() if v.get("init_error") is not None}
        ro = {int(k): (v["prob_meas1_prep0"], v["prob_meas0_prep1"]) for k, v in q.items()
              if v.get("prob_meas1_prep0") is not None}
        for b in "XZ":
            tasks.append((str(j.folder), b, a.channel, a.leak, a.strength, a.spread, a.shots, hmm, re, ro))
    del all_jobs
    cost = {"r1": 4, "d5": 3, "d3": 1}
    tasks.sort(key=lambda t: -cost[Path(t[0]).name[-2:]])
    counts: dict = defaultdict(lambda: defaultdict(dict))
    t0 = time.time()
    # one task per worker process, so that each job's circuits are released after use
    with mp.Pool(a.procs, maxtasksperchild=1) as pool:
        for i, (name, basis, byr) in enumerate(pool.imap_unordered(_task, tasks, chunksize=1)):
            for rounds, k in byr.items():
                for p, v in k.items():
                    counts[name][basis].setdefault(p, {})[rounds] = v
            print(f"{i + 1}/{len(tasks)} {name} {basis} [{time.time() - t0:.0f}s]", flush=True)
    orig = json.loads((DIAG / "results/rerun_model-fit-bitflip_cz1.5_d1_p0.5.json").read_text())
    out = {"params": vars(a), "rounds": list(ROUNDS), "jobs": {}}
    for n in names:
        leak, k_out = {}, {}
        for b in "XZ":
            leak[b], k_out[b] = {}, {}
            for p, byr in counts[n][b].items():
                k = [byr[r] for r in ROUNDS]
                f = fit_per_round(ROUNDS, [x / a.shots for x in k], a.shots)
                leak[b][p] = [f.per_round, f.uncertainty]
                k_out[b][p] = {"k": k, "binomial": F.binomial_point(ROUNDS, k, a.shots)}
        out["jobs"][n] = {"model": orig["jobs"][n]["model"], "chip": orig["jobs"][n]["chip"], "leak": leak, "counts": k_out}
    path = HERE / f"rerun_{a.tag}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path.relative_to(DIAG.parent.parent), f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
