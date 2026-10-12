"""Per-round logical-error counts for every hardware curve, and the stored model predictions.

Reads each run folder's analysis.json (prespecified decode) and, as an independent check,
re-decodes the memory shots from shots.npz with the repo's own decoder (the stim model
of each translated circuit at the job's recorded decoupled_dephasing f, then pymatching),
exactly as heavyhex.experiment.run.analyze does. Nothing in src/ or runs/ is modified.

    python docs/diagnosis/revision/refits/extract_counts.py [--no-redecode]

Writes docs/diagnosis/revision/refits/counts.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


from heavyhex.experiment import run as R
from heavyhex.experiment.circuits import ROUNDS, SETTINGS
from heavyhex.patches.placement import Calibration

REPO = Path(__file__).resolve().parents[4]
OUT = Path(__file__).resolve().parent / "counts.json"

# Shots behind each stored prediction (run.py: SIMULATED_SHOTS, SIDE_SHOTS).
PREDICTIONS = {
    "predicted_at_measured_dephasing": R.SIMULATED_SHOTS,
    "predicted_from_calibration_after": R.SIMULATED_SHOTS,
    "predicted_with_each_qubits_dephasing": R.SIDE_SHOTS,
}


def memory_rows(settings: list[dict], basis: str) -> list[dict]:
    return [r for s, r in zip(SETTINGS, settings) if s.kind == "memory" and s.basis == basis]


def redecode(folder: Path, f: float) -> dict[str, dict[str, list[int]]]:
    """Failures per (basis, patch, round count), re-decoded from the saved shots."""
    run = json.loads((folder / "run.json").read_text())
    patches = R.patches_of(run)
    calibration = Calibration.load(folder / "calibration.json")
    circuits = R.load_circuits(folder)
    shots = R.load_shots(folder)
    found: dict = {b: {p: [] for p in patches} for b in "XZ"}
    for i, (setting, circuit) in enumerate(zip(SETTINGS, circuits)):
        if setting.kind != "memory":
            continue
        model, records, _ = R.simulated(setting, circuit, calibration, run["dt"], f)
        measurements = R.in_record_order(shots[i], records)
        fired, flips = model.compile_m2d_converter().convert(
            measurements=measurements, separate_observables=True
        )
        failed = R._decode(model, fired) != flips
        for k, p in enumerate(patches):
            found[setting.basis][p].append(int(failed[:, k].sum()))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-redecode", action="store_true")
    args = ap.parse_args()
    out: dict = {"rounds": list(ROUNDS), "jobs": {}}
    for folder in sorted((REPO / "runs").glob("ibm_fez-*")):
        t0 = time.time()
        run = json.loads((folder / "run.json").read_text())
        a = json.loads((folder / "analysis.json").read_text())
        patches = a["patches"]
        shots = R._memory_shots(R.load_shots(folder))
        job: dict = {
            "day": run["rep"]["day"],
            "rep": run["rep"]["rep"],
            "pairing": run["rep"]["pairing"],
            "patches": patches,
            "shots": shots,
            "f": a["decoupled_dephasing"],
            "f_uncertainty": a["idle_test"]["median_uncertainty"],
            "observed": {},
            "registered_fit": a["observed"]["fits"],
            "predicted": {},
        }
        for b in "XZ":
            rows = memory_rows(a["observed"]["settings"], b)
            job["observed"][b] = {
                p: [int(round(r["logical_error"][k] * shots)) for r in rows]
                for k, p in enumerate(patches)
            }
        sources = dict(PREDICTIONS)
        frozen = json.loads((folder / "prediction.json").read_text())
        for name, n in sources.items():
            pred = a.get(name)
            if not pred:
                continue
            job["predicted"][name] = {"shots": n, "fits": pred["fits"], "P": {}}
            for b in "XZ":
                rows = memory_rows(pred["settings"], b)
                job["predicted"][name]["P"][b] = {
                    p: [r["logical_error"][k] for r in rows] for k, p in enumerate(patches)
                }
        for f_end, pred in frozen.items():
            name = f"frozen_f{f_end}"
            job["predicted"][name] = {"shots": R.SIMULATED_SHOTS, "fits": pred["fits"], "P": {}}
            for b in "XZ":
                rows = memory_rows(pred["settings"], b)
                job["predicted"][name]["P"][b] = {
                    p: [r["logical_error"][k] for r in rows] for k, p in enumerate(patches)
                }
        if not args.no_redecode:
            again = redecode(folder, a["decoupled_dephasing"])
            job["redecoded"] = again
            job["redecode_matches"] = again == job["observed"]
        out["jobs"][folder.name] = job
        print(f"{folder.name}: {shots} shots, redecode matches: {job.get('redecode_matches')}"
              f" ({time.time() - t0:.0f} s)", flush=True)
    OUT.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
