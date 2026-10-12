"""Re-derive a job's stored prediction with the frozen code (Data availability).

Before each job, prepare() stored predictions at f = 0 and f = 1 in prediction.json, from
200,000 simulated shots per circuit with fixed seeds. This runs run.predict again on the stored
circuits and calibration, with the same shot count and seeds, and compares every number with
the stored one: how many are equal, and the largest difference.

A few minutes and about 1 GB per value of f.

    python docs/analysis/repredict.py [--job ibm_fez-2026-10-04-r1] [--workers 2]   # writes results/repredict.json
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor

from heavyhex.experiment import run as R
from heavyhex.patches.placement import Calibration

from common import RUNS, WORKERS, read, write_json


def leaves(x, path=()):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, (*path, k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, (*path, i))
    else:
        yield path, x


def again(task: tuple) -> tuple:
    job, f = task
    folder = RUNS / job
    t0 = time.time()
    out = R.predict(
        R.load_circuits(folder),
        Calibration.load(folder / "calibration.json"),
        read(folder / "run.json")["dt"],
        decoupled_dephasing=f,
    )
    return task, (json.loads(json.dumps(out)), round(time.time() - t0, 1))


def compare(new: dict, old: dict) -> dict:
    a, b = dict(leaves(new)), dict(leaves(old))
    assert a.keys() == b.keys(), "the re-derived prediction has a different layout"
    numbers = [k for k in a if isinstance(a[k], float) or isinstance(b[k], float)]
    differ = [k for k in a if a[k] != b[k]]
    diffs = [abs(a[k] - b[k]) for k in numbers if a[k] != b[k]]
    return {
        "values": len(a),
        "floats": len(numbers),
        "unequal": len(differ),
        "largest_abs_difference": max(diffs, default=0.0),
        "first_unequal": [list(map(str, k)) for k in differ[:5]],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--job", default="ibm_fez-2026-10-04-r1")
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    stored = read(RUNS / a.job / "prediction.json")
    tasks = [(a.job, float(f)) for f in stored]
    with ProcessPoolExecutor(a.workers) as pool:
        done = dict(pool.map(again, tasks))
    out: dict = {"job": a.job, "shots_per_circuit": R.SIMULATED_SHOTS, "f": {}}
    for (_, f), (new, seconds) in done.items():
        out["f"][str(f)] = compare(new, stored[str(f)]) | {"seconds": seconds}
        print(f"f = {f}: {out['f'][str(f)]}")
    write_json("repredict.json", out)


if __name__ == "__main__":
    main()
