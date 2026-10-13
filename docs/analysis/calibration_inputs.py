"""Two medians of IBM's calibration that the paper quotes (Sec. model, Sec. leakage).

  single_qubit   by day, the median single-qubit (SX) error over the chip's working qubits, and
                 its ratio to the median CZ error (runs/*-r1/calibration.json)
  readout_gap    Pr(0|1) - Pr(1|0) over the flag and relay qubits (the 38 of results/hmm.json),
                 one value per qubit and day where IBM's split is recorded for the calibration
                 in force (properties.readout_split: days 1-3); a reset that flips a qubit only
                 when it reads 1 fails more often after a 1 by about this much
  t1_flags_relays  the reported T1 of the flag or relay qubit of each of the 320 Z-memory series
                 of results/hmm.json, from its job's calibration (a transmon's |2> decays in
                 about T1/2)

Reads committed files only; a second.

    python docs/analysis/calibration_inputs.py      # writes results/calibration_inputs.json
"""

from __future__ import annotations

import statistics

from heavyhex.patches.placement import BROKEN

from common import RUNS, read, result, write_json
from properties import readout_split

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
    series = result("hmm.json")
    readout_qubits = sorted({row["qubit"] for row in series})
    gaps = []
    for day in DAYS:
        split = readout_split(day)
        gaps += [split[q][1] - split[q][0] for q in readout_qubits if q in split]
    out["readout_gap"] = {
        "qubits": len(readout_qubits),
        "values": len(gaps),
        "days": "qubit-days with the split for the calibration in force (properties.py)",
        "median": statistics.median(gaps),
        "quartiles": statistics.quantiles(gaps, n=4)[::2],
    }
    t1 = {}
    for job in {row["job"] for row in series}:
        t1[job] = read(RUNS / f"ibm_fez-{job}" / "calibration.json")["t1_us"]
    t1_series = [t1[row["job"]][str(row["qubit"])] for row in series]
    out["t1_flags_relays"] = {
        "series": len(t1_series),
        "median_us": statistics.median(t1_series),
        "quartiles_us": statistics.quantiles(t1_series, n=4)[::2],
    }
    write_json("calibration_inputs.json", out)
    print(out)


if __name__ == "__main__":
    main()
