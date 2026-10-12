"""Spot check on real memory circuits: frame simulator (original, fixed) against the exact
tableau simulation (CZ left out; CZ -> random phase), at the paper's fitted parameters.

    python spot_exact.py JOB BASIS ROUNDS SHOTS
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DIAG = HERE.parents[1]
sys.path[:0] = [str(HERE), str(DIAG)]
import leaky as fixed  # noqa: E402
import tableau_ref  # noqa: E402
import rerun  # noqa: E402
from common import ROUND_US, jobs, memory_index  # noqa: E402
from fit_leak2 import stats  # noqa: E402
from heavyhex.experiment import run as R  # noqa: E402

spec = importlib.util.spec_from_file_location("leaky_orig", DIAG / "leaky.py")
orig = importlib.util.module_from_spec(spec); sys.modules["leaky_orig"] = orig; spec.loader.exec_module(orig)


def main():
    name, basis, rounds, shots = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    job = next(j for j in jobs() if j.name == name)
    _, _, life, read = rerun.leakage_params()
    q = rerun.PROPS[job.day]["qubits"]
    kw = dict(leak=rerun._AllQubits(defaultdict(lambda: 0.002)), lifetime_us=life, half_round_us=ROUND_US / 2,
              read_leaked=read, partner_phase=0.25, partner_kind="bitflip", transfer=0.05,
              reset_error={int(k): v["init_error"] for k, v in q.items() if v.get("init_error") is not None},
              readout={int(k): (v["prob_meas1_prep0"], v["prob_meas0_prep1"]) for k, v in q.items()
                       if v.get("prob_meas1_prep0") is not None})
    model, records, labels = job.model(memory_index(basis, rounds))
    runs = {}
    t = time.time()
    runs["frame original"] = orig.simulate(model, 20000, orig.Leakage(**kw), seed=11)
    runs["frame fixed (dephase)"] = fixed.simulate(model, 20000, fixed.Leakage(**kw), seed=11)
    runs["exact dephase"] = tableau_ref.simulate(model, shots, fixed.Leakage(**kw), "dephase", seed=12)
    runs["exact identity (CZ left out)"] = tableau_ref.simulate(model, shots, fixed.Leakage(**kw), "identity", seed=13)
    out = {}
    for k, (det, obs) in runs.items():
        failed = R._decode(model, det) != obs
        st = stats(det, labels)
        out[k] = {"shots": len(det), "logical_error": failed.mean(0).tolist(),
                  "logical_se": np.sqrt(failed.mean(0) * (1 - failed.mean(0)) / len(det)).tolist(),
                  "det_mean": float(det.mean()), **st}
        print(f"{k:30s} n={len(det):6d} logical {np.round(failed.mean(0), 4)}  det {det.mean():.4f}  " +
              "  ".join(f"{s} {v:.4g}" for s, v in st.items()), flush=True)
    print("elapsed", time.time() - t)
    (HERE / f"spot_{name}_{basis}{rounds}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
