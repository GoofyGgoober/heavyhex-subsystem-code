"""Table "rerun" of the paper under one estimator: device, calibration model and fitted leakage
model (fit-bitflip), each fitted with the prespecified weighted-log fit and with the binomial MLE.

    python docs/diagnosis/revision/refits/rerun_refit.py

Reads rerun_per_round.json (rerun_per_round.py) and refits.json (refit.py); writes
rerun_refit.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import fitlib as F

HERE = Path(__file__).resolve().parent


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    rr = json.loads((HERE / "rerun_per_round.json").read_text())
    ref = json.loads((HERE / "refits.json").read_text(encoding="utf-8"))
    rounds, N = tuple(rr["rounds"]), rr["shots"]
    eps: dict = {}  # (rep key, variant, basis, d) -> {wls, binomial}
    for name, j in rr["jobs"].items():
        key = f"{j['day']} r{j['rep']}"
        for v, by_b in j["failures"].items():
            for b, by_p in by_b.items():
                for p, k in by_p.items():
                    w = F.wls(rounds, np.array(k) / N, N)
                    m = F.binomial_point(rounds, k, N)
                    eps[(key, v, b, p)] = {"wls": w.eps, "binomial": m, "k": k}
    keys = sorted(ref["per_rep"])
    table: dict = {}
    for src in ("device", *rr["variants"]):
        for est in ("wls", "binomial"):
            col = {}
            for b in "XZ":
                e3, e5 = [], []
                for k in keys:
                    if src == "device":
                        x = ref["per_rep"][k][b]
                        e3.append(x["eps3"]["wls"][0] if est == "wls" else x["eps3"]["binomial"])
                        e5.append(x["eps5"]["wls"][0] if est == "wls" else x["eps5"]["binomial"])
                    else:
                        e3.append(eps[(k, src, b, "3")][est])
                        e5.append(eps[(k, src, b, "5")][est])
                e3, e5 = np.array(e3), np.array(e5)
                col[f"{b} eps3"] = float(e3.mean())
                col[f"{b} eps5"] = float(e5.mean())
                col[f"{b} lambda"] = float((e3 / e5).mean())
            table[f"{src} {est}"] = col
    out = {"shots_per_circuit_simulated": N, "means_over_ten_reps": table,
           "per_curve": {" ".join(map(str, k)): v for k, v in eps.items()},
           "regenerated_matches_stored_eps_to": rr["max_abs_difference_from_stored_eps"]}
    (HERE / "rerun_refit.json").write_text(json.dumps(out, indent=1) + "\n")
    rows = ["X eps3", "X eps5", "Z eps3", "Z eps5", "X lambda", "Z lambda"]
    cols = list(table)
    print("quantity  " + "  ".join(f"{c:>22s}" for c in cols))
    for r in rows:
        s = 100 if "eps" in r else 1
        print(f"{r:9s} " + "  ".join(f"{s * table[c][r]:22.3f}" for c in cols))
    print("regenerated P(n) reproduce the stored eps to", rr["max_abs_difference_from_stored_eps"])


if __name__ == "__main__":
    main()
