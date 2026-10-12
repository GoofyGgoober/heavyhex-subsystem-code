"""Two medians of IBM's calibration that the paper quotes (Sec. model, Sec. leakage).

  single_qubit   by day, the median single-qubit (SX) error over the chip's working qubits, and
                 its ratio to the median CZ error (runs/*-r1/calibration.json)
  readout_gap    Pr(0|1) - Pr(1|0) over the flag and relay qubits (the 38 of results/hmm.json),
                 one value per qubit and day, from results/properties.json; a reset that flips a
                 qubit only when it reads 1 fails more often after a 1 by about this much

Reads committed files only; a second.

    python docs/analysis/calibration_inputs.py      # writes results/calibration_inputs.json
"""

from __future__ import annotations

import statistics

from heavyhex.patches.placement import BROKEN

from common import RUNS, read, result, write_json

DAYS = ("2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08")


def main() -> None:
    out: dict = {"single_qubit": {}, "readout_gap": {}}
    for day in DAYS:
        cal = read(RUNS / f"ibm_fez-{day}-r1" / "calibration.json")
        sx = [v for v in cal["single_qubit"].values() if v is not None and v < BROKEN]
        cz = [v for v in cal["cz"].values() if v is not None and v < BROKEN]
        out["single_qubit"][day] = {
            "median": statistics.median(sx),
            "median_cz": statistics.median(cz),
            "cz_over_single_qubit": statistics.median(cz) / statistics.median(sx),
        }
    readout_qubits = sorted({row["qubit"] for row in result("hmm.json")})
    props = result("properties.json")
    gaps = []
    for day in DAYS:
        qubits = props[day]["qubits"]
        for q in readout_qubits:
            p = qubits[str(q)]
            gaps.append(p["prob_meas0_prep1"] - p["prob_meas1_prep0"])
    out["readout_gap"] = {
        "qubits": len(readout_qubits),
        "values": len(gaps),
        "median": statistics.median(gaps),
        "quartiles": statistics.quantiles(gaps, n=4)[::2],
    }
    write_json("calibration_inputs.json", out)
    print(out)


if __name__ == "__main__":
    main()
