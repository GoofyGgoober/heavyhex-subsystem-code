"""Side check: the chip's shots decoded with the per-qubit-f model's matching weights.

The prespecified analysis decodes the chip's shots with the model at the job's median f;
the stored per-qubit prediction decodes its simulated shots with the per-qubit model.
This decodes the chip's memory shots both ways, with the repo's own simulated(),
in_record_order() and _decode(), and fits eps and Lambda with the repo's _fits(). The
median-f decode must reproduce the prespecified observed fits exactly (it is deterministic).
Moderate: a few minutes per worker-job.

    python docs/analysis/perqubit_f_decode.py [--workers 2]     # writes results/perqubit_f_decode.json
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

from common import RESULTS, WORKERS, job_folders, read


def decode_job(folder: str) -> tuple[str, dict]:
    from heavyhex.experiment import run as R
    from heavyhex.experiment.circuits import SETTINGS
    from heavyhex.patches.placement import Calibration

    path = Path(folder)
    info, analysis = read(path / "run.json"), read(path / "analysis.json")
    calibration = Calibration.load(path / "calibration.json")
    circuits = R.load_circuits(path)
    shots = R.load_shots(path)
    f = analysis["decoupled_dephasing"]
    own = {row["qubit"]: row["dephasing"] for row in analysis["dephasing_by_qubit"]}
    patches = analysis["patches"]
    out = {}
    for version, byq in (("median_f", None), ("per_qubit_f", own)):
        rows = []
        for i, setting in enumerate(SETTINGS):
            if not setting.decoded:
                rows.append({})
                continue
            model, records, _ = R.simulated(setting, circuits[i], calibration, info["dt"], f, byq)
            measured = R.in_record_order(shots[i], records)
            fired, flips = model.compile_m2d_converter().convert(
                measurements=measured, separate_observables=True
            )
            failed = R._decode(model, fired) != flips
            rows.append({"logical_error": failed.mean(axis=0).tolist()})
        out[version] = R._fits(rows, R._memory_shots(shots), patches)["fits"]
    prespecified = analysis["observed"]["fits"]
    out["median_f_reproduces_prespecified"] = all(
        abs(out["median_f"][b]["per_round"][p][0] - prespecified[b]["per_round"][p][0]) < 1e-12
        for b in "XZ"
        for p in patches
    )
    return path.name, out


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    with Pool(a.workers) as pool:
        found = dict(pool.map(decode_job, [str(f) for f in job_folders()]))
    (RESULTS / "perqubit_f_decode.json").write_text(json.dumps(found, indent=1) + "\n")
    for name, row in found.items():
        print(
            name,
            "median-f decode reproduces prespecified:",
            row["median_f_reproduces_prespecified"],
        )


if __name__ == "__main__":
    main()
