"""Independent verification of the decoder audit: cache the repo's stim models and the chip's detectors.

For every job (15) and memory setting (12), build the stim model exactly as the
prespecified analysis does (heavyhex.experiment.run.simulated with the job's
calibration.json, run.json dt and analysis.json decoupled_dephasing), and turn the
chip's saved shots into detector bits / observable flips with that model's m2d
converter. Saved under cache/ so the verification scripts do not rebuild models.

No QPU use; reads runs/ only.
"""

from __future__ import annotations

import json
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
CACHE = HERE / "cache"
sys.path[:0] = [str(REPO / "src")]


def build(folder_name: str) -> str:
    import numpy as np
    from heavyhex.experiment import run as R
    from heavyhex.experiment.circuits import SETTINGS
    from heavyhex.patches.placement import Calibration

    folder = REPO / "runs" / folder_name
    run = json.loads((folder / "run.json").read_text())
    analysis = json.loads((folder / "analysis.json").read_text())
    cal = Calibration.load(folder / "calibration.json")
    circuits = R.load_circuits(folder)
    shots = R.load_shots(folder)
    f = analysis["decoupled_dephasing"]
    out = CACHE / folder_name
    out.mkdir(parents=True, exist_ok=True)
    meta = {"folder": folder_name, "rep": run["rep"], "f": f, "dt": run["dt"],
            "patches": analysis["patches"], "settings": {}}
    for i, s in enumerate(SETTINGS):
        if s.kind != "memory":
            continue
        model, records, labels = R.simulated(s, circuits[i], cal, run["dt"], f)
        (out / f"model_{i}.stim").write_text(str(model))
        measured = R.in_record_order(shots[i], records)
        det, obs = model.compile_m2d_converter().convert(measurements=measured, separate_observables=True)
        np.savez_compressed(out / f"chip_{i}.npz", det=det, obs=obs)
        meta["settings"][i] = {"basis": s.basis, "rounds": s.rounds, "labels": labels,
                               "saved_logical_error": analysis["observed"]["settings"][i]["logical_error"],
                               "shots": int(det.shape[0])}
        with open(out / f"records_{i}.pkl", "wb") as fh:
            pickle.dump(records, fh)
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    return folder_name


if __name__ == "__main__":
    folders = sorted(p.name for p in (REPO / "runs").glob("ibm_fez-2026-10-*-r*") if (p / "analysis.json").exists())
    print(folders)
    with ProcessPoolExecutor(6) as ex:
        for name in ex.map(build, folders):
            print("done", name, flush=True)
