"""Medians and ages of IBM's calibration that the paper quotes (Sec. model, Sec. leakage).

  single_qubit   by day, the median single-qubit (SX) error over the chip's working qubits, its
                 ratio to the median CZ error (runs/*-r1/calibration.json), and the model's noise
                 on the two SX gates around a CZ, 2 (3/2) r_q, over the CZ's own (5/4) r_qq'
  age            by day, how old the snapshot in force (calibration.json) and IBM's previous
                 snapshot (results/properties.json, `served`) were when IBM created the day's
                 first job, and how many of the in-force T1, T2, CZ and readout values equal
                 the previous snapshot's
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
from datetime import datetime

from heavyhex.patches.placement import BROKEN

from common import RUNS, read, result, write_json
from properties import readout_split

DAYS = ("2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08")


def minutes(later: str, earlier: str) -> float:
    return (datetime.fromisoformat(later) - datetime.fromisoformat(earlier)).total_seconds() / 60


def age(day: str, previous: dict) -> dict:
    """The day's snapshot in force and IBM's previous one, against the day's first job."""
    cal = read(RUNS / f"ibm_fez-{day}-r1" / "calibration.json")
    first = min(read(job)["created_utc"] for job in RUNS.glob(f"ibm_fez-{day}-*/job.json"))
    qubits, couplers = previous["qubits"], previous["cz"]
    same: dict[str, list[int]] = {}
    for name, ours, theirs, scale in (
        ("t1", "t1_us", "T1", 1e6),
        ("t2", "t2_us", "T2", 1e6),
        ("readout", "readout", "readout_error", 1.0),
    ):
        pairs = [
            (value, qubits[q][theirs] * scale)
            for q, value in cal[ours].items()
            if value is not None and qubits.get(q, {}).get(theirs) is not None
        ]
        same[name] = [sum(abs(a - b) <= 1e-9 * max(1.0, abs(a)) for a, b in pairs), len(pairs)]
    pairs = []
    for edge, value in cal["cz"].items():
        flipped = "-".join(edge.split("-")[::-1])
        theirs = couplers.get(edge, couplers.get(flipped))
        if value is not None and theirs is not None:
            pairs.append((value, theirs))
    same["cz"] = [sum(abs(a - b) <= 1e-12 for a, b in pairs), len(pairs)]
    return {
        "first_job_created_utc": first,
        "in_force_minutes_old": minutes(first, cal["calibrated"]),
        "previous_minutes_old": minutes(first, previous["served"]),
        "equal_to_previous [equal, compared]": same,
    }


def main() -> None:
    out: dict = {"single_qubit": {}, "age": {}, "readout_gap": {}}
    snapshots = result("properties.json")
    for day in DAYS:
        cal = read(RUNS / f"ibm_fez-{day}-r1" / "calibration.json")
        sx = [v for v in cal["single_qubit"].values() if v is not None and v < BROKEN]
        cz = [v for v in cal["cz"].values() if v is not None and v < BROKEN]
        r_q, r_qq = statistics.median(sx), statistics.median(cz)
        out["single_qubit"][day] = {
            "median": r_q,
            "median_cz": r_qq,
            "cz_over_single_qubit": r_qq / r_q,
            "sx_pair_noise_over_cz_noise": 2 * 1.5 * r_q / (1.25 * r_qq),
        }
        out["age"][day] = age(day, snapshots[day])
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
