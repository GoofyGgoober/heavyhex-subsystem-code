"""Every job's memory circuits rerun with the Pauli-frame simulator and decoded as the chip's
shots were (the calibration model's error model at the job's measured f; 10,000 shots per
circuit, seed = round count). Lambda and eps are fitted as the prespecified analysis fits them.

`variants` runs named models (frame_sim.py, the left-out CZ as "reference"):

  model       the calibration model as prespecified (the analysis' prediction, resampled)
  leak        + leakage: each qubit's own leak per CZ where the flags and relays measured it
              (hmm_causes.py), the median elsewhere; the fitted lifetime and leaked readout
  leak+ro     + each reset's own error (IBM's init_error, where the model used the readout
              error) and each readout's own P(1|0) and P(0|1) in place of their average
  leak+ro+cz  + every CZ's error times cz_scale
  fit         the model of leakage_frame_grid.py: every qubit leaks 0.2% per CZ, a leaked
              qubit's CZ gives its partner a random Pauli a quarter of the time and spreads
              to it 5% of the time; IBM's reset and readout errors
  fit-bitflip the same with an X on the partner (an exchange of excitation), the better fit

    python docs/analysis/leakage_frame_rerun.py variants --variants model fit-bitflip --out leakage_frame_rerun_bitflip.json
    python docs/analysis/leakage_frame_rerun.py variants --variants model fit --out leakage_frame_rerun_depolarize.json
    python docs/analysis/leakage_frame_rerun.py variants --variants model leak leak+ro --jobs 2026-10-08-r1 --out leakage_frame_rerun_leak_ro.json

The last tests IBM's reset and readout errors inside the leakage model on one job; it was
written by an earlier version of this script that did not record the knobs. readout_ibm.py
tests them in the calibration model on all fifteen jobs.

`point TAG` reruns the fitted model at one parameter point with the left-out CZ handled by
--mode ("reference", "dephase", or exactly by tableau_ref.py: "exact-identity",
"exact-dephase"), and takes the chip and the calibration model from
leakage_frame_rerun_bitflip.json (leakage off, so the CZ handling does not touch them):

    python docs/analysis/leakage_frame_rerun.py point dephase_paperparams --mode dephase
    python docs/analysis/leakage_frame_rerun.py point dephase_refit --mode dephase --leak 0.0005
    python docs/analysis/leakage_frame_rerun.py point exactidentity_paperparams --mode exact-identity
    python docs/analysis/leakage_frame_rerun.py point verify_reference --mode reference --jobs 2026-10-08-r2-d3

Heavy: 10,000 shots x 12 circuits x 15 jobs per variant (the exact modes take many hours).
Writes results/leakage_frame_rerun_<TAG>.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from collections import defaultdict

import numpy as np

from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import ROUNDS
from heavyhex.simulation.noisy import fit_per_round

import frame_sim
from common import RESULTS, ROUND_US, WORKERS, jobs, memory_index
from leakage import AllQubits, hmm_inputs, ibm_errors


def leakage_params(cz_scale: float = 1.0):
    """Each job's measured leak per CZ (hmm_causes.json), their median, the lifetime and leaked readout."""
    why = json.loads((RESULTS / "hmm_causes.json").read_text())
    per = defaultdict(dict)
    for w in why:
        if np.isfinite(w["leak_per_cz"]):
            per[w["job"]][w["qubit"]] = w["leak_per_cz"]
    median_cz = float(np.median([v for d in per.values() for v in d.values()]))
    life, read = hmm_inputs()
    return per, median_cz, life, read


# Not measured by the flags and relays: the leak rate of the other qubits (data and Z
# ancillas) as a share of the readout qubits' median, and how often a leaked qubit's CZ
# dephases its partner.
KNOBS = {"data_scale": 1.0, "partner_phase": 0.5}
# The fitted leakage model (leakage_frame_grid.py), best for both partner kinds.
FIT = {"leak": 0.002, "strength": 0.25, "spread": 0.05}


def variant(job, name: str, per, median_cz, life, read, cz_scale) -> frame_sim.Leakage | None:
    if name == "model":
        return None
    if name in ("fit", "fit-bitflip"):
        reset, readout = ibm_errors(job.day)
        return frame_sim.Leakage(
            leak=AllQubits(defaultdict(lambda: FIT["leak"])),
            lifetime_us=life,
            half_round_us=ROUND_US / 2,
            read_leaked=read,
            partner_phase=FIT["strength"],
            partner_kind="bitflip" if name == "fit-bitflip" else "depolarize",
            transfer=FIT["spread"],
            reset_error=reset,
            readout=readout,
        )
    own = per.get(job.name, {})
    leak = defaultdict(lambda: median_cz * KNOBS["data_scale"], own)
    lk = frame_sim.Leakage(
        lifetime_us=life,
        half_round_us=ROUND_US / 2,
        read_leaked=read,
        partner_phase=KNOBS["partner_phase"],
        cz_scale=cz_scale if name.endswith("+cz") else 1.0,
    )
    lk.leak = AllQubits(leak)
    if "+ro" in name:
        lk.reset_error, lk.readout = ibm_errors(job.day)
    return lk


def run_job(job, variants, shots, params: tuple[dict, float, float, float], cz_scale):
    out = {}
    for v in variants:
        errors = {b: defaultdict(list) for b in "XZ"}
        for basis in "XZ":
            for rounds in ROUNDS:
                idx = memory_index(basis, rounds)
                model, _, labels = job.model(idx)
                det, obs = frame_sim.simulate(
                    model, shots, variant(job, v, *params, cz_scale), seed=rounds
                )
                failed = R._decode(model, det) != obs
                patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
                for k, patch in enumerate(patches):
                    errors[basis][patch].append(failed[:, k].mean())
        fits = {b: {p: fit_per_round(ROUNDS, e, shots) for p, e in errors[b].items()} for b in "XZ"}
        out[v] = {b: {p: [f.per_round, f.uncertainty] for p, f in fits[b].items()} for b in "XZ"}
    obs = job.analysis["observed"]["fits"]
    out["chip"] = {b: {p: obs[b]["per_round"][p] for p in obs[b]["per_round"]} for b in "XZ"}
    return out


def variants_main(a) -> None:
    KNOBS.update(data_scale=a.data_scale, partner_phase=a.partner_phase)
    params = leakage_params()
    print(
        f"leak per CZ median {params[1]:.4f}, lifetime {params[2]:.0f} us, leaked reads 1 with {params[3]:.2f}",
        flush=True,
    )
    results = {}
    for job in jobs():
        if a.jobs and job.name not in a.jobs:
            continue
        results[job.name] = run_job(job, a.variants, a.shots, params, a.cz_scale)
        r = results[job.name]
        print(
            job.name,
            " | ".join(
                f"{b} d{p}: chip {100 * r['chip'][b][p][0]:.1f} "
                + " ".join(f"{v} {100 * r[v][b][p][0]:.1f}" for v in a.variants)
                for b in "XZ"
                for p in r["chip"][b]
            ),
            flush=True,
        )
    tag = (
        "-".join(a.variants).replace("+", "_")
        + f"_cz{a.cz_scale:g}_d{a.data_scale:g}_p{a.partner_phase:g}"
    )
    path = RESULTS / (a.out or f"leakage_frame_rerun_{tag}.json")
    path.write_text(
        json.dumps(
            {
                "cz_scale": a.cz_scale,
                "knobs": KNOBS,
                "shots": a.shots,
                "params": params[1:],
                "jobs": results,
            },
            indent=1,
        )
        + "\n"
    )
    print("wrote", path)


# ------------------------------------------------------------------ one point, the CZ handled by --mode
def _job(args):
    name, kind, mode, leak, strength, spread, shots = args
    job = next(j for j in jobs() if j.name == name)
    life, read = hmm_inputs()
    reset, readout = ibm_errors(job.day)
    lk = frame_sim.Leakage(
        leak=AllQubits(defaultdict(lambda: leak)),
        lifetime_us=life,
        half_round_us=ROUND_US / 2,
        read_leaked=read,
        partner_phase=strength,
        partner_kind=kind,
        transfer=spread,
        reset_error=reset,
        readout=readout,
        skipped_cz="dephase" if mode.startswith("exact-") else mode,
    )
    errors = {b: defaultdict(list) for b in "XZ"}
    for basis in "XZ":
        for rounds in ROUNDS:
            model, _records, labels = job.model(memory_index(basis, rounds))
            if mode.startswith("exact-"):
                import tableau_ref

                det, obs = tableau_ref.simulate(
                    model, shots, lk, mode.removeprefix("exact-"), seed=rounds
                )
            else:
                det, obs = frame_sim.simulate(model, shots, lk, seed=rounds)
            failed = R._decode(model, det) != obs
            patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
            for k, patch in enumerate(patches):
                errors[basis][patch].append(failed[:, k].mean())
    fits = {b: {p: fit_per_round(ROUNDS, e, shots) for p, e in errors[b].items()} for b in "XZ"}
    return name, {b: {p: [f.per_round, f.uncertainty] for p, f in fits[b].items()} for b in "XZ"}


def point_main(a) -> None:
    orig = json.loads((RESULTS / "leakage_frame_rerun_bitflip.json").read_text())
    names = a.jobs or list(orig["jobs"])
    with mp.Pool(min(a.workers, len(names))) as pool:
        res = dict(
            pool.map(
                _job,
                [(n, a.kind, a.mode, a.leak, a.strength, a.spread, a.shots) for n in names],
                chunksize=1,
            )
        )
    params = {
        k: getattr(a, k)
        for k in ("tag", "kind", "mode", "leak", "strength", "spread", "shots", "jobs")
    }
    out = {"params": params, "jobs": {}}
    for n in names:
        out["jobs"][n] = {
            "model": orig["jobs"][n]["model"],
            "chip": orig["jobs"][n]["chip"],
            "leak": res[n],
        }
    path = RESULTS / f"leakage_frame_rerun_{a.tag}.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="command", required=True)
    v = sub.add_parser("variants", help="named models, the left-out CZ as 'reference'")
    v.add_argument("--variants", nargs="*", default=["model", "leak", "leak+ro"])
    v.add_argument("--shots", type=int, default=10000)
    v.add_argument("--cz-scale", type=float, default=1.5)
    v.add_argument("--data-scale", type=float, default=KNOBS["data_scale"])
    v.add_argument("--partner-phase", type=float, default=KNOBS["partner_phase"])
    v.add_argument("--jobs", nargs="*")
    v.add_argument("--out", help="results file name (default from the variants and knobs)")
    p = sub.add_parser("point", help="the fitted model at one point, the CZ handled by --mode")
    p.add_argument("tag")
    p.add_argument("--kind", default="bitflip")
    p.add_argument(
        "--mode",
        default="dephase",
        choices=["reference", "dephase", "exact-identity", "exact-dephase"],
    )
    p.add_argument("--leak", type=float, default=0.002)
    p.add_argument("--strength", type=float, default=0.25)
    p.add_argument("--spread", type=float, default=0.05)
    p.add_argument("--shots", type=int, default=10000)
    p.add_argument("--jobs", nargs="*")
    p.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    (variants_main if a.command == "variants" else point_main)(a)


if __name__ == "__main__":
    main()
