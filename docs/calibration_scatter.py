"""How far errors in IBM's CZ, readout and √X numbers move the f read off X memory.

The allowance CALIBRATION_SCATTER_F in run.py comes from here. Two of IBM's calibrations
a day apart stand in for a calibration and the chip it is wrong about: the chip is
simulated with one day's CZ, readout and √X errors, the model with the other day's
(T₁, T₂ and the patches from the model's day), both at f = 0.5. The model then reads
f off the chip's ε₃ and stabilizer detector rates, by the curve through its own
predictions at f = 0, 0.5 and 1, as analyze does. Both directions are run.

The preview used IBM's calibration of 3 October 2026 against the day before's:

    git show 9414ac1:runs/ibm_fez-2026-10-02-offline/calibration.json > before.json
    python docs/calibration_scatter.py runs/ibm_fez-2026-10-03-offline/calibration.json before.json

Local simulation only; it uses no QPU time.
"""

from __future__ import annotations

import argparse
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

from heavyhex.experiment import run
from heavyhex.experiment.circuits import (
    DISTANCES,
    ROUNDS,
    SETTINGS,
    for_fez,
    logical_circuit,
    with_durations,
)
from heavyhex.patches.placement import Calibration, fez_qubits
from heavyhex.simulation.noisy import fit_per_round

TRUE_F = 0.5
CURVE = (0.0, TRUE_F, 1.0)
X_MEMORY = [i for i, s in enumerate(SETTINGS) if s.kind == "memory" and s.basis == "X"]


def simulate(model_day: Path, chip_day: Path | None, f: float, shots: int) -> tuple:
    """ε₃ and the stabilizer detector rates of X memory, on the model day's patches."""
    from qiskit_ibm_runtime.fake_provider import FakeFez

    model = Calibration.load(model_day)
    noise = model
    if chip_day:
        other = Calibration.load(chip_day)
        noise = replace(model, cz=other.cz, readout=other.readout, single_qubit=other.single_qubit)
    target = FakeFez().target
    if model.durations_ns:
        target = with_durations(target, model.durations_ns)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        chips = {str(d): fez_qubits(d, model) for d in DISTANCES}
    errors, rates = [], []
    for i in X_MEMORY:
        setting = SETTINGS[i]
        circuit = for_fez(logical_circuit(setting, chips, target.num_qubits)[0], target)
        stim_circuit, _, labels = run.simulated(setting, circuit, noise, target.dt, f)
        sampler = stim_circuit.compile_detector_sampler(seed=setting.rounds)
        detectors, flips = sampler.sample(shots, separate_observables=True)
        failed = run._decode(stim_circuit, detectors) != flips
        stabilizers = [k for k, (_, _, name) in enumerate(labels) if " " not in name]
        errors.append(float(failed[:, 0].mean()))
        rates.append(detectors[:, stabilizers].mean(axis=0))
    return fit_per_round(ROUNDS, errors, shots).per_round, np.concatenate(rates)


def read_off(model_day: Path, chip_day: Path, shots: int, pool: ProcessPoolExecutor) -> dict:
    """The f the model day reads off a chip with the other day's errors, at a true f of 0.5."""
    jobs = [(model_day, None, f, shots) for f in CURVE] + [(model_day, chip_day, TRUE_F, shots)]
    *curve, (eps_chip, rates_chip) = pool.map(simulate, *zip(*jobs))
    grid = np.linspace(-0.5, 2.0, 2501)
    eps_curve = np.polyfit(CURVE, [eps for eps, _ in curve], 2)
    from_eps = float(grid[np.argmin(np.abs(np.polyval(eps_curve, grid) - eps_chip))])
    rate_curves = np.polyfit(CURVE, np.array([r for _, r in curve]), 2)
    misfit = [((np.polyval(rate_curves, g) - rates_chip) ** 2).sum() for g in grid]
    return {
        "eps3": [eps for eps, _ in curve],
        "eps3_chip": eps_chip,
        "from_eps3": from_eps,
        "from_detector_rates": float(grid[int(np.argmin(misfit))]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("day", type=Path, help="one calibration file")
    parser.add_argument("other_day", type=Path, help="another day's calibration file")
    parser.add_argument("--shots", type=int, default=run.SIMULATED_SHOTS, metavar="N")
    args = parser.parse_args(argv)
    with ProcessPoolExecutor(4) as pool:
        for model_day, chip_day in ((args.day, args.other_day), (args.other_day, args.day)):
            r = read_off(model_day, chip_day, args.shots, pool)
            print(
                f"model {model_day}, chip errors from {chip_day}: ε₃ at f = 0, 0.5, 1 "
                + ", ".join(f"{e:.2%}" for e in r["eps3"])
                + f"; chip {r['eps3_chip']:.2%}. f read off ε₃ {r['from_eps3']:.2f}, "
                f"off detector rates {r['from_detector_rates']:.2f} (true {TRUE_F})"
            )


if __name__ == "__main__":
    main()
