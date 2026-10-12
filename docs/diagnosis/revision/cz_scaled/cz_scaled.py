"""No-leakage calibration model with every CZ error scaled, against the leakage model and the chip.

Backs (or refutes) the paper's sentence in "Why the model misses": "Logical errors largely
follow detection rates: on one job, a model without leakage but with three times the CZ
error gives similar logical errors." No result on file backed it; this reruns it.

Jobs: the four the leakage fit used (fit_leak2.FIT_JOBS: days 3 and 5). Variants, each
simulated with leaky.py and decoded exactly as the chip's shots were (pymatching on the
calibration model's own error model at the job's measured f), as rerun.py does:

  model   the calibration model (rerun.py "model")
  fit     the fitted leakage model of the paper's Table (rerun.py "fit")
  czK     the calibration model without leakage, every CZ's depolarizing error times K
          (leaky.Leakage(cz_scale=K) with no leak; only DEPOLARIZE2 follows CZ in the model)

The error per round is fitted as the prespecified analysis fits it (fit_per_round); the chip's
comes from each job's analysis.json. Seeds follow rerun.py (seed = number of rounds), so
"model" and "fit" reproduce docs/diagnosis/results/rerun_model-fit_cz1.5_d1_p0.5.json.

Also records detection rates per patch for the 8-round circuits (stabilizer detectors of
rounds 1-7, as fit_leak2.stats counts them; flags and relays separately), for the device
and every variant, to check whether logical errors follow detection rates.

Exploratory, post hoc. Runs locally with stim/numpy; no QPU.

    python docs/diagnosis/revision/cz_scaled/cz_scaled.py [--shots 10000] [--scales 1.5 2 3]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # docs/diagnosis

from common import jobs, memory_index  # noqa: E402
import leaky  # noqa: E402
import rerun  # noqa: E402
from heavyhex.experiment import run as R  # noqa: E402
from heavyhex.experiment.circuits import ROUNDS  # noqa: E402
from heavyhex.simulation.noisy import fit_per_round  # noqa: E402

# Same as docs/diagnosis/fit_leak2.py FIT_JOBS (not imported: that module reads sys.argv).
FIT_JOBS = ("2026-10-08-r1", "2026-10-08-r2-d5", "2026-10-08-r2-d3", "2026-10-06-r1")
LONGEST = max(ROUNDS)


def make_variant(job, name, params):
    if name == "model":
        return None
    if name == "fit":
        return rerun.variant(job, "fit", *params, 1.0)
    if name.startswith("cz"):
        return leaky.Leakage(cz_scale=float(name[2:]))  # no leak: leak dict empty
    raise ValueError(name)


def detection_rates(det: np.ndarray, labels, rounds: int) -> dict:
    """Per patch: mean rate of bulk stabilizer detectors and of flag/relay detectors."""
    out = {}
    for patch in dict.fromkeys(str(p) for p, _, _ in labels):
        stab = [c for c, (p, r, n) in enumerate(labels) if str(p) == patch and " " not in n and 1 <= r <= rounds - 1]
        flag = [c for c, (p, r, n) in enumerate(labels) if str(p) == patch and " " in n]
        out[patch] = {
            "stab": float(det[:, stab].mean()) if stab else None,
            "flag": float(det[:, flag].mean()) if flag else None,
        }
    return out


def run_job(job, variants, shots, params):
    errors = {v: {b: defaultdict(list) for b in "XZ"} for v in variants}
    rates = {v: {} for v in ["chip", *variants]}
    for basis in "XZ":
        for rounds in ROUNDS:
            idx = memory_index(basis, rounds)
            model, records, labels = job.model(idx)
            patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
            if rounds == LONGEST:
                chip_det, _ = job.chip_detectors(idx, model, records)
                rates["chip"][basis] = detection_rates(chip_det, labels, rounds)
            for v in variants:
                det, obs = leaky.simulate(model, shots, make_variant(job, v, params), seed=rounds)
                failed = R._decode(model, det) != obs
                for k, patch in enumerate(patches):
                    errors[v][basis][patch].append(float(failed[:, k].mean()))
                if rounds == LONGEST:
                    rates[v][basis] = detection_rates(det, labels, rounds)
    out = {"eps": {}, "raw": errors, "rates": rates}
    for v in variants:
        out["eps"][v] = {
            b: {p: [fit_per_round(ROUNDS, e, shots).per_round, fit_per_round(ROUNDS, e, shots).uncertainty]
                for p, e in errors[v][b].items()}
            for b in "XZ"
        }
    obs = job.analysis["observed"]["fits"]
    out["eps"]["chip"] = {b: {p: obs[b]["per_round"][p] for p in obs[b]["per_round"]} for b in "XZ"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--scales", nargs="*", type=float, default=[1.5, 2.0, 3.0])
    ap.add_argument("--jobs", nargs="*", default=list(FIT_JOBS))
    ap.add_argument("--out", default=str(HERE / "cz_scaled.json"))
    args = ap.parse_args()
    variants = ["model", "fit", *[f"cz{s:g}" for s in args.scales]]
    params = rerun.leakage_params()
    results = {}
    for job in jobs():
        if job.name not in args.jobs:
            continue
        t0 = time.time()
        results[job.name] = run_job(job, variants, args.shots, params)
        e = results[job.name]["eps"]
        print(job.name, f"({time.time() - t0:.0f} s)", flush=True)
        for b in "XZ":
            for p in e["chip"][b]:
                print(f"  {b} d{p}: " + "  ".join(f"{v} {100 * e[v][b][p][0]:.2f}" for v in ["chip", *variants]), flush=True)
    Path(args.out).write_text(json.dumps(
        {"shots": args.shots, "scales": args.scales, "variants": variants, "rounds": list(ROUNDS),
         "fit_params": rerun.FIT, "leak_params": params[1:], "jobs": results}, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
