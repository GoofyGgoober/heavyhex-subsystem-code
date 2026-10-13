"""The calibration model with IBM's own reset and readout errors.

The model gives every reset and every readout the calibration's readout error, the mean of
P(1|0) and P(0|1), as a symmetric flip. IBM also reports a reset error (init_error), several
times smaller, and the two readout errors separately (properties.py says which are recorded).
For every job and both bases this samples each memory circuit three times: as the model has
it ("model"); with each reset's own init_error where IBM reports one and each readout
flipping a 0 with P(1|0) and a 1 with P(0|1) where those are recorded for the calibration in
force ("ibm"); and with the init_errors only ("ibm_reset"). All are decoded with the model's
priors and fitted with the prespecified fit. For the 8-round circuits it also compares each
detector's rate with the device's. readout.py summarizes the result.

About 6 minutes (10,000 shots per circuit).

    python docs/analysis/readout_ibm.py [--shots 10000]   # writes results/readout_ibm.json
"""

from __future__ import annotations

import argparse

import numpy as np
import stim

from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import ROUNDS
from heavyhex.simulation.noisy import fit_per_round

from common import RESULTS, jobs, memory_index, write_json
from leakage import ibm_errors


def split_readout(model: stim.Circuit, reset_error: dict | None):
    """The model without its readout flips, which come back per measurement as (qubits, flips),
    and with each reset's flip taken from reset_error where it has the qubit."""
    out = stim.Circuit()
    qubits: list[int] = []
    flips: list[float] = []
    reset: set[int] = set()
    for ins in model.flattened():
        just_reset, reset = reset, set()
        name, args, targets = ins.name, ins.gate_args_copy(), ins.targets_copy()
        if name == "X_ERROR" and reset_error is not None:
            for t in targets:
                p = reset_error.get(t.value, args[0]) if t.value in just_reset else args[0]
                out.append("X_ERROR", [t.value], [p])
        elif name == "M":
            out.append("M", targets)
            qubits += [t.value for t in targets]
            flips += [args[0] if args else 0.0] * len(targets)
        else:
            out.append(ins)
            if name == "R":
                reset = {t.value for t in targets}
    return out, np.array(qubits), np.array(flips)


def sample(model: stim.Circuit, shots: int, seed: int, reset_error=None, readout=None):
    """Detection events and observable flips. Readout flips act on the true outcome: with
    readout, a 0 flips with P(1|0) and a 1 with P(0|1); without, both with the model's error."""
    circuit, qubits, flips = split_readout(model, reset_error)
    true = circuit.compile_sampler(seed=seed).sample(shots)
    if readout is None:
        p = np.broadcast_to(flips, true.shape)
    else:
        p10 = np.array([readout.get(q, (f, f))[0] for q, f in zip(qubits, flips)])
        p01 = np.array([readout.get(q, (f, f))[1] for q, f in zip(qubits, flips)])
        p = np.where(true, p01[None, :], p10[None, :])
    read = true ^ (np.random.default_rng(seed + 7).random(true.shape) < p)
    return circuit.compile_m2d_converter().convert(measurements=read, separate_observables=True)


def detector_rates(rates: np.ndarray, device: np.ndarray) -> dict:
    """Mean rate, and the RMS of each detector's deviation relative to the device's rate."""
    return {
        "mean": float(rates.mean()),
        "rms_relative_to_device": float(np.sqrt(np.mean(((rates - device) / device) ** 2))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    parser.add_argument("--shots", type=int, default=10000)
    args = parser.parse_args()
    out: dict = {
        "shots": args.shots,
        "variants": {
            "model": "as run: each reset and readout flips with the calibration's readout error",
            "ibm": "resets flip with IBM's init_error, readouts with P(1|0) and P(0|1)",
            "ibm_reset": "resets flip with IBM's init_error, readouts as in the model",
        },
        "jobs": {},
    }
    longest = ROUNDS[-1]
    for job in jobs():
        reset_error, readout = ibm_errors(job.day)
        variants = {
            "model": {},
            "ibm": {"reset_error": reset_error, "readout": readout},
            "ibm_reset": {"reset_error": reset_error},
        }
        found = {}
        for basis in "XZ":
            failed: dict = {v: {} for v in variants}
            rates = {}
            for rounds in ROUNDS:
                idx = memory_index(basis, rounds)
                model, records, labels = job.model(idx)
                patches = list(dict.fromkeys(str(p) for p, _, _ in labels))
                for v, kw in variants.items():
                    det, obs = sample(model, args.shots, rounds, **kw)
                    wrong = R._decode(model, det) != obs
                    for k, patch in enumerate(patches):
                        failed[v].setdefault(patch, []).append(float(wrong[:, k].mean()))
                    if rounds == longest:
                        rates[v] = det.mean(0)
                if rounds == longest:
                    rates["device"] = job.chip_detectors(idx, model, records)[0].mean(0)
            eps = {}
            for patch in failed["model"]:
                fits = {v: fit_per_round(ROUNDS, failed[v][patch], args.shots) for v in variants}
                eps[patch] = {v: [f.per_round, f.uncertainty] for v, f in fits.items()}
            device = np.maximum(rates["device"], 1e-4)
            found[basis] = {
                "eps": eps,
                f"detectors_{longest}_rounds": {
                    "device_mean": float(device.mean()),
                    **{v: detector_rates(rates[v], device) for v in variants},
                },
            }
        out["jobs"][job.name] = found
        print(
            job.name,
            "  ".join(
                f"{b} d{p} {100 * e['model'][0]:.2f}->{100 * e['ibm'][0]:.2f}%"
                for b in "XZ"
                for p, e in found[b]["eps"].items()
            ),
            flush=True,
        )
    write_json("readout_ibm.json", out)
    print("wrote", RESULTS / "readout_ibm.json")


if __name__ == "__main__":
    main()
