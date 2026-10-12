"""How far the per-qubit-f prediction moves under the idle test's own uncertainty in f.

The prespecified test's f term is (dQ/df) x sigma_f for the single job-wide median f.
The per-qubit prediction has no such term in the repo. This script estimates one by
Monte Carlo, for X memory only (Z memory does not depend on the data qubits' f):

  each draw, every measured data qubit gets max(share_q + sigma_q z_q, 0) (share_q the
  raw idle-test estimate, sigma_q its standard error, Appendix C's clip at zero), and the
  ancillas, flags and the data qubits the idle test could not pin (Appendix C exclusion)
  get max(median + sigma_median z_0, 0); the X-memory circuits are then simulated with
  the repo's own simulated()/_decode() at SIDE_SHOTS (50,000) shots and the same seeds
  as the prespecified side prediction (common random numbers, so the spread over draws is
  the spread due to f, not shot noise), and eps3, eps5 and Lambda are fitted with the
  repo's fit_per_round. One extra evaluation at the point estimates ("central") checks
  that the stored per-qubit prediction is reproduced within shot noise.

    python docs/diagnosis/revision/perqubit_f/mc_from_f.py [draws] [workers]

Writes mc_from_f.json next to this file. Exploratory; nothing prespecified depends on it.
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_pq import HERE, job_folders, read  # noqa: E402

DRAWS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
WORKERS = int(sys.argv[2]) if len(sys.argv) > 2 else 14
SEED = 20261010

_cache: dict = {}


def _job(folder: str):
    if folder not in _cache:
        from heavyhex.experiment import run as R
        from heavyhex.patches.placement import Calibration

        path = Path(folder)
        _cache[folder] = (
            read(path / "run.json"),
            read(path / "analysis.json"),
            Calibration.load(path / "calibration.json"),
            R.load_circuits(path),
        )
    return _cache[folder]


def f_values(analysis: dict, rng: np.random.Generator | None):
    """(f for every qubit not in the table, {chip qubit: f}) for one draw; rng None = point estimate."""
    idle = analysis["idle_test"]
    median, sigma_median = idle["median"], idle["median_uncertainty"]
    if rng is None:
        base = max(analysis["decoupled_dephasing"], 0.0)
        return base, {row["qubit"]: row["dephasing"] for row in analysis["dephasing_by_qubit"]}
    base = max(median + sigma_median * rng.standard_normal(), 0.0)
    own = {}
    for row in analysis["dephasing_by_qubit"]:
        q = int(row["data"][1:]) - 1
        raw = idle["per_qubit"][str(row["patch"])][q]
        if row["measured"]:
            own[row["qubit"]] = max(raw + row["uncertainty"] * rng.standard_normal(), 0.0)
        else:
            own[row["qubit"]] = base  # excluded qubit: the median, as the repo does
    return base, own


def evaluate(task):
    folder, k = task
    from heavyhex.experiment import run as R
    from heavyhex.experiment.circuits import ROUNDS, SETTINGS
    from heavyhex.simulation.noisy import fit_per_round

    info, analysis, calibration, circuits = _job(folder)
    rng = None if k < 0 else np.random.default_rng([SEED, zlib.crc32(Path(folder).name.encode()), k])
    base, own = f_values(analysis, rng)
    patches = analysis["patches"]
    errors = {p: [] for p in patches}
    for i, setting in enumerate(SETTINGS):
        if not (setting.kind == "memory" and setting.basis == "X"):
            continue
        model, _, _ = R.simulated(setting, circuits[i], calibration, info["dt"], base, own)
        sampler = model.compile_detector_sampler(seed=setting.rounds)
        detectors, flips = sampler.sample(R.SIDE_SHOTS, separate_observables=True)
        failed = R._decode(model, detectors) != flips
        for j, p in enumerate(patches):
            errors[p].append(float(failed[:, j].mean()))
    out = {"k": k, "base_f": base}
    for p in patches:
        fit = fit_per_round(ROUNDS, errors[p], R.SIDE_SHOTS)
        out[f"eps{p}"] = fit.per_round
        out[f"eps{p}_uncertainty"] = fit.uncertainty
    if "3" in patches and "5" in patches:
        out["lambda"] = out["eps3"] / out["eps5"]
    return Path(folder).name, out


def main() -> None:
    folders = [str(f) for f in job_folders()]
    tasks = [(f, k) for f in folders for k in range(-1, DRAWS)]
    start = time.time()
    results: dict = {Path(f).name: {"central": None, "draws": []} for f in folders}
    with Pool(WORKERS) as pool:
        for n, (name, row) in enumerate(pool.imap_unordered(evaluate, tasks), 1):
            if row["k"] < 0:
                results[name]["central"] = row
            else:
                results[name]["draws"].append(row)
            if n % 25 == 0:
                print(f"{n}/{len(tasks)} done, {time.time() - start:.0f} s", flush=True)
    for name, entry in results.items():
        entry["draws"].sort(key=lambda r: r["k"])
        keys = [k for k in ("eps3", "eps5", "lambda") if k in entry["draws"][0]]
        entry["sd"] = {k: float(np.std([r[k] for r in entry["draws"]], ddof=1)) for k in keys}
        entry["mean"] = {k: float(np.mean([r[k] for r in entry["draws"]])) for k in keys}
    payload = {
        "draws": DRAWS,
        "shots": 50_000,
        "seed": SEED,
        "what": "SD over draws of the X-memory per-qubit-f prediction, f_q ~ N(share_q, sigma_q) "
        "clipped at 0, unmeasured qubits ~ N(median, sigma_median) clipped at 0; common random numbers",
        "jobs": results,
        "seconds": time.time() - start,
    }
    (HERE / "mc_from_f.json").write_text(json.dumps(payload, indent=1) + "\n")
    print(f"wrote mc_from_f.json in {time.time() - start:.0f} s")


if __name__ == "__main__":
    main()
