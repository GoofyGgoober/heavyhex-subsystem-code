"""Rerun every job's memory circuits with what the diagnosis found, and set them against the chip.

Variants, each simulated with leaky.py and decoded exactly as the chip's shots were (the
calibration model's own error model, at the job's measured f):

  model     the calibration model as registered (the analysis' prediction, resampled)
  leak      + leakage: each qubit's own leak per CZ where the flags and relays measured it,
            the median elsewhere; the fitted lifetime and the leaked readout
  leak+ro   + each reset's own error (IBM's init_error, where the model used the readout
            error) and each readout's own P(1|0) and P(0|1) in place of their average
  leak+ro+cz  + every CZ's error times cz_scale
  fit       the model fitted to detection statistics in fit_leak2.py: every qubit leaks
            0.2% per CZ, a leaked qubit's CZ gives its partner a random Pauli a quarter of
            the time and spreads to it 5% of the time; IBM's reset and readout errors
  fit-bitflip  the same, with the partner's disturbance a bit flip (an exchange of
            excitation), the better fit to the detection statistics

Λ and ε are fitted exactly as the registered analysis fits them. Exploratory.

    python docs/diagnosis/rerun.py [--shots 10000] [--cz-scale 1.5] [--jobs 2026-10-08-r1 ...]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

from common import REPO, ROUND_US, jobs, memory_index
from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import ROUNDS
from heavyhex.simulation.noisy import fit_per_round
import leaky


def leakage_params(cz_scale: float = 1.0):
    hmm = json.loads((REPO / "docs/diagnosis/results/leakage_hmm.json").read_text())
    why = json.loads((REPO / "docs/diagnosis/results/why.json").read_text())
    per = defaultdict(dict)
    for w in why:
        if np.isfinite(w["leak_per_cz"]):
            per[w["job"]][w["qubit"]] = w["leak_per_cz"]
    median_cz = float(np.median([v for d in per.values() for v in d.values()]))
    life = float(np.median([1 / max(r["chip"]["b"], 1e-6) for r in hmm])) * ROUND_US
    read = float(np.median([r["chip"]["e_L"] for r in hmm]))
    return per, median_cz, life, read


PROPS = json.loads((REPO / "docs/diagnosis/results/properties.json").read_text())


# Not measured by the flags and relays, so fitted to the detector rates (fit_leak.py):
# the leak rate of the other qubits (data and Z ancillas) as a share of the readout
# qubits' median, and how often a leaked qubit's CZ dephases its partner.
KNOBS = {"data_scale": 1.0, "partner_phase": 0.5}
# The fitted leakage model (fit_leak2.py): every qubit leaks with the same probability per
# CZ, and a leaked qubit's CZ gives its partner a random Pauli or spreads to it.
FIT = {"leak": 0.002, "strength": 0.25, "spread": 0.05}  # best for both partner kinds


def variant(job, name: str, per, median_cz, life, read, cz_scale) -> leaky.Leakage | None:
    if name == "model":
        return None
    if name in ("fit", "fit-bitflip"):
        q = PROPS[job.day]["qubits"]
        return leaky.Leakage(
            leak=_AllQubits(defaultdict(lambda: FIT["leak"])), lifetime_us=life, half_round_us=ROUND_US / 2,
            read_leaked=read, partner_phase=FIT["strength"],
            partner_kind="bitflip" if name == "fit-bitflip" else "depolarize", transfer=FIT["spread"],
            reset_error={int(k): v["init_error"] for k, v in q.items() if v.get("init_error") is not None},
            readout={int(k): (v["prob_meas1_prep0"], v["prob_meas0_prep1"]) for k, v in q.items()
                     if v.get("prob_meas1_prep0") is not None})
    own = per.get(job.name, {})
    leak = defaultdict(lambda: median_cz * KNOBS["data_scale"], own)
    lk = leaky.Leakage(lifetime_us=life, half_round_us=ROUND_US / 2, read_leaked=read,
                       partner_phase=KNOBS["partner_phase"],
                       cz_scale=cz_scale if name.endswith("+cz") else 1.0)
    lk.leak = _AllQubits(leak)
    if "+ro" in name:
        q = PROPS[job.day]["qubits"]
        lk.reset_error = {int(k): v["init_error"] for k, v in q.items() if v.get("init_error") is not None}
        lk.readout = {int(k): (v["prob_meas1_prep0"], v["prob_meas0_prep1"]) for k, v in q.items()
                      if v.get("prob_meas1_prep0") is not None}
    return lk


class _AllQubits(dict):
    """Every qubit has a leak rate: its own if measured, the median otherwise."""

    def __init__(self, leak):
        super().__init__()
        self._leak = leak

    def get(self, q, default=None):
        return self._leak[q]

    def __bool__(self):
        return True


def run_job(job, variants, shots, params, cz_scale):
    out = {}
    for v in variants:
        errors = {b: defaultdict(list) for b in "XZ"}
        for basis in "XZ":
            for rounds in ROUNDS:
                idx = memory_index(basis, rounds)
                model, records, labels = job.model(idx)
                det, obs = leaky.simulate(model, shots, variant(job, v, *params, cz_scale), seed=rounds)
                failed = R._decode(model, det) != obs
                patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
                for k, patch in enumerate(patches):
                    errors[basis][patch].append(failed[:, k].mean())
        fits = {b: {p: fit_per_round(ROUNDS, e, shots) for p, e in errors[b].items()} for b in "XZ"}
        out[v] = {b: {p: [f.per_round, f.uncertainty] for p, f in fits[b].items()} for b in "XZ"}
    obs = job.analysis["observed"]["fits"]
    out["chip"] = {b: {p: obs[b]["per_round"][p] for p in obs[b]["per_round"]} for b in "XZ"}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", type=int, default=10000)
    ap.add_argument("--cz-scale", type=float, default=1.5)
    ap.add_argument("--jobs", nargs="*")
    ap.add_argument("--variants", nargs="*", default=["model", "leak", "leak+ro"])
    ap.add_argument("--data-scale", type=float, default=KNOBS["data_scale"])
    ap.add_argument("--partner-phase", type=float, default=KNOBS["partner_phase"])
    args = ap.parse_args()
    KNOBS.update(data_scale=args.data_scale, partner_phase=args.partner_phase)
    params = leakage_params()
    print(f"leak per CZ median {params[1]:.4f}, lifetime {params[2]:.0f} us, leaked reads 1 with {params[3]:.2f}", flush=True)
    results = {}
    for job in jobs():
        if args.jobs and job.name not in args.jobs:
            continue
        results[job.name] = run_job(job, args.variants, args.shots, params, args.cz_scale)
        r = results[job.name]
        print(job.name, " | ".join(
            f"{b} d{p}: chip {100*r['chip'][b][p][0]:.1f} " + " ".join(f"{v} {100*r[v][b][p][0]:.1f}" for v in args.variants)
            for b in "XZ" for p in r["chip"][b]), flush=True)
    tag = "-".join(args.variants).replace("+", "_") + f"_cz{args.cz_scale:g}_d{args.data_scale:g}_p{args.partner_phase:g}"
    path = REPO / f"docs/diagnosis/results/rerun_{tag}.json"
    path.write_text(json.dumps({"cz_scale": args.cz_scale, "knobs": KNOBS, "shots": args.shots, "params": params[1:], "jobs": results}, indent=1) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
