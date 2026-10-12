"""Where the day-to-day swing of placement 1's predicted Λ_X comes from (Sec. days).

Pair 1 ran on the same qubits every day, and its predicted Λ_X moved between 0.54 (day 1)
and 0.82 (day 2) while the measured value stayed constant. Here the frozen run.predict is run
again on those two days' jobs with some reported values replaced by their median over the
other four days' calibrations: the readout error, T1, T2 and single-qubit error of the chosen
qubits, and the CZ error of every coupler with both ends among them. The circuits are the
stored ones and f is the job's measured value, as in the tested prediction.

  as reported   the calibration of the day, unchanged
  d3, d5, both  every qubit of the distance-3 patch, the distance-5 patch, or both
  q...          the named chip qubits only

Every variant samples with the same seeds (run.predict seeds each circuit by its round count),
so the differences between variants carry less noise than the values. 100,000 shots per circuit
by default, about a minute and 0.5 GB per variant.

    python docs/analysis/day_swing.py [--shots 100000] [--workers 2]   # writes results/day_swing.json
"""

from __future__ import annotations

import argparse
import dataclasses
import statistics
import time
from concurrent.futures import ProcessPoolExecutor

from heavyhex.experiment import run as R
from heavyhex.patches.placement import Calibration

from common import RUNS, WORKERS, day_number, read, write_json

DAYS = ("2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08")
VARIANTS = {
    "2026-10-05": [
        ("as reported", []),
        ("d3", ["d3"]),
        ("d5", ["d5"]),
        ("both", ["d3", "d5"]),
        ("q4", [4]),
        ("q5 q25", [5, 25]),
        ("q4 q5 q24 q25", [4, 5, 24, 25]),
    ],
    "2026-10-04": [
        ("as reported", []),
        ("d3", ["d3"]),
        ("d5", ["d5"]),
        ("both", ["d3", "d5"]),
    ],
}
FIELDS = ("readout", "t1_us", "t2_us", "single_qubit")


def folder(day: str):
    return RUNS / f"ibm_fez-{day}-r1"


def with_medians(day: str, qubits: set[int]) -> tuple[Calibration, list]:
    """The day's calibration with the qubits' values set to their median over the other days."""
    cal = Calibration.load(folder(day) / "calibration.json")
    others = [Calibration.load(folder(d) / "calibration.json") for d in DAYS if d != day]

    def median(field: str, key):
        values = [getattr(c, field).get(key) for c in others if getattr(c, field)]
        values = [v for v in values if v is not None]
        return statistics.median(values) if values else None

    changes = []
    fields: dict = {}
    for name in FIELDS:
        values = dict(getattr(cal, name) or {})
        for q in sorted(qubits):
            m = median(name, q)
            if q in values and m is not None:
                changes.append([name, [q], values[q], m])
                values[q] = m
        fields[name] = values
    cz = dict(cal.cz)
    for pair, value in cal.cz.items():
        m = median("cz", pair)
        if set(pair) <= qubits and value is not None and m is not None:
            changes.append(["cz", list(pair), value, m])
            cz[pair] = m
    fields["cz"] = cz
    return dataclasses.replace(cal, **fields), changes


def variant(task: tuple) -> tuple:
    day, _, spec, shots = task
    info = read(folder(day) / "run.json")
    qubits: set[int] = set()
    for s in spec:
        qubits |= set(info["chips"][s[1]]) if isinstance(s, str) else {s}
    cal, changes = with_medians(day, qubits)
    f = read(folder(day) / "analysis.json")["decoupled_dephasing"]
    t0 = time.time()
    out = R.predict(
        R.load_circuits(folder(day)), cal, info["dt"], decoupled_dephasing=f, shots=shots
    )
    fx = out["fits"]["X"]
    # the changes that moved a value by more than half, largest first
    big = sorted(
        (c for c in changes if abs(c[2] - c[3]) > 0.5 * min(abs(c[2]), abs(c[3]))),
        key=lambda c: -abs(c[2] - c[3]) / max(min(abs(c[2]), abs(c[3])), 1e-12),
    )
    return task, {
        "qubits": len(qubits),
        "values_changed": len(changes),
        "changed_by_more_than_half [field, qubits, reported, median of other days]": big,
        "eps3": fx["per_round"]["3"][0],
        "eps5": fx["per_round"]["5"][0],
        "lambda": fx["lambda"],
        "lambda_uncertainty": fx.get("lambda_uncertainty"),
        "seconds": round(time.time() - t0, 1),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--shots", type=int, default=100_000)
    ap.add_argument("--workers", type=int, default=WORKERS)
    a = ap.parse_args()
    tasks = [(day, name, tuple(spec), a.shots) for day, vs in VARIANTS.items() for name, spec in vs]
    with ProcessPoolExecutor(a.workers) as pool:
        done = dict(pool.map(variant, tasks))
    out: dict = {"shots_per_circuit": a.shots, "placement": 1, "days": {}}
    for day, vs in VARIANTS.items():
        rows = {name: done[(day, name, tuple(spec), a.shots)] for name, spec in vs}
        base = rows["as reported"]["lambda"]
        for row in rows.values():
            row["lambda_change"] = row["lambda"] - base
        out["days"][day] = {"day": day_number(day), "variants": rows}
        for name, row in rows.items():
            print(
                f"day {day_number(day)} {name:14s} Λ {row['lambda']:.3f} ({row['lambda_change']:+.3f})"
            )
    write_json("day_swing.json", out)


if __name__ == "__main__":
    main()
