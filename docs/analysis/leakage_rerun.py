"""Table rerun's leakage column with an exact simulation of the leakage channel.

Simulates every job's memory circuits (both bases, every round count) at the given leakage
parameters with exact_sim.py, decodes them as leakage_frame_rerun.py does (R._decode, the
calibration model's priors; seed = round count; 10,000 shots per circuit), and stores the
logical-failure counts at every round count, so that eps can be fitted both with the
prespecified weighted-log fit (heavyhex.simulation.noisy.fit_per_round) and by binomial
likelihood (fitting.py). Leakage settings as in leakage_frame_rerun.py's fit-bitflip
variant: every qubit leaks with the same probability per CZ; partner X error and transfer
as given; the hidden Markov lifetime and leaked readout; each day's IBM reset errors and
P(1|0), P(0|1). The chip and the calibration model come from leakage_frame_rerun_bitflip.json.

    python docs/analysis/leakage_rerun.py TAG --channel identity --leak 0.002 --strength 0.25 --spread 0.05
           [--shots 10000] [--workers 2] [--jobs ...]

The committed reruns (leakage_fit.py reads them):

    identity_paperpoint  --channel identity --leak 0.002   --strength 0.25  --spread 0.05
    identity_extbest     --channel identity --leak 0.002   --strength 0.175 --spread 0.075
    identity_alt         --channel identity --leak 0.0015  --strength 0.25  --spread 0.075
    dephase_refit        --channel dephase  --leak 0.0001  --strength 0.5   --spread 0.3
    dephase_alt          --channel dephase  --leak 0.00025 --strength 0.5   --spread 0.15
    pi_refit             --channel pi       --leak 0.0005  --strength 0.25  --spread 0.15

Heavy: hours per rerun. Writes results/leakage_rerun_<TAG>.json.
"""

from __future__ import annotations

import os

# one BLAS thread per process (OpenBLAS otherwise commits about 0.5 GB per process)
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import multiprocessing as mp
import sys
import time
from collections import defaultdict
from pathlib import Path

from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import ROUNDS
from heavyhex.simulation.noisy import fit_per_round

import exact_sim as E
import fitting as F
from common import REPO, RESULTS, WORKERS, Job, jobs, memory_index
from leakage import ibm_errors
from leakage_grid import hmm_params


def _task(args):
    """One job and basis, every round count (seed = round count, as leakage_frame_rerun.py)."""
    folder, basis, channel, leak, strength, spread, shots, hmm, reset_error, readout = args
    job = Job(Path(folder))
    lk = E.Leakage(
        leak=E.Uniform(leak),
        lifetime_us=hmm[0],
        half_round_us=hmm[2],
        read_leaked=hmm[1],
        partner_phase=strength,
        partner_kind="bitflip",
        transfer=spread,
        reset_error=reset_error,
        readout=readout,
    )
    out = {}
    for rounds in ROUNDS:
        model, _records, labels = job.model(memory_index(basis, rounds))
        det, obs = E.simulate(model, shots, lk, channel, seed=rounds)
        failed = R._decode(model, det) != obs
        patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
        out[rounds] = {p: int(failed[:, k].sum()) for k, p in enumerate(patches)}
    return job.name, basis, out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("tag")
    ap.add_argument("--channel", choices=["identity", "dephase", "pi"], default="identity")
    ap.add_argument("--leak", type=float, default=0.002)
    ap.add_argument("--strength", type=float, default=0.25)
    ap.add_argument("--spread", type=float, default=0.05)
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--workers", type=int, default=WORKERS)
    ap.add_argument("--jobs", nargs="*")
    a = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")
    hmm = hmm_params()
    all_jobs = [j for j in jobs() if not a.jobs or j.name in a.jobs]
    names = [j.name for j in all_jobs]
    tasks = []
    for j in all_jobs:
        re, ro = ibm_errors(j.day)
        for b in "XZ":
            tasks.append(
                (str(j.folder), b, a.channel, a.leak, a.strength, a.spread, a.shots, hmm, re, ro)
            )
    del all_jobs
    cost = {"r1": 4, "d5": 3, "d3": 1}
    tasks.sort(key=lambda t: -cost[Path(t[0]).name[-2:]])
    counts: dict = defaultdict(lambda: defaultdict(dict))
    t0 = time.time()
    # one task per worker process, so that each job's circuits are released after use
    with mp.Pool(a.workers, maxtasksperchild=1) as pool:
        for i, (name, basis, byr) in enumerate(pool.imap_unordered(_task, tasks, chunksize=1)):
            for rounds, k in byr.items():
                for p, v in k.items():
                    counts[name][basis].setdefault(p, {})[rounds] = v
            print(f"{i + 1}/{len(tasks)} {name} {basis} [{time.time() - t0:.0f}s]", flush=True)
    orig = json.loads((RESULTS / "leakage_frame_rerun_bitflip.json").read_text())
    params = {k: v for k, v in vars(a).items() if k != "workers"}
    out = {"params": params, "rounds": list(ROUNDS), "jobs": {}}
    for n in names:
        leak, k_out = {}, {}
        for b in "XZ":
            leak[b], k_out[b] = {}, {}
            for p, byr in counts[n][b].items():
                k = [byr[r] for r in ROUNDS]
                f = fit_per_round(ROUNDS, [x / a.shots for x in k], a.shots)
                leak[b][p] = [f.per_round, f.uncertainty]
                k_out[b][p] = {"k": k, "binomial": F.binomial_point(ROUNDS, k, a.shots)}
        out["jobs"][n] = {
            "model": orig["jobs"][n]["model"],
            "chip": orig["jobs"][n]["chip"],
            "leak": leak,
            "counts": k_out,
        }
    path = RESULTS / f"leakage_rerun_{a.tag}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path.relative_to(REPO), f"[{time.time() - t0:.0f}s]")


if __name__ == "__main__":
    main()
