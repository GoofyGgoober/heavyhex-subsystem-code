"""IBM's per-qubit readout and reset figures that the run folders do not keep.

runs/*/calibration.json keeps only what the model uses: CZ, readout error (the mean of P(1|0)
and P(0|1)), T1, T2 and single-qubit error. Some diagnoses also use each qubit's two readout
errors P(1|0) and P(0|1) and its initialization error, which only IBM's full properties carry.

results/properties.json records them, one IBM snapshot per day. It was written by an earlier
version of this script, which asked for the properties at the calibration's own timestamp;
IBM serves the latest snapshot strictly older than the time asked, so each day's entry is the
snapshot before the one the jobs ran under ("served", 36 to 84 minutes before "calibrated").
Its readout errors equal the ones in force on all 156 qubits on days 1-3 and on one qubit on
each of days 4 and 5, where the readout had been calibrated again in between. The analyses
therefore use a qubit's recorded figures only where its recorded readout error is the one in
force (runs/*/calibration.json), taking the initialization error to come with the readout
calibration; elsewhere every reset and readout keeps the readout error in force, as in the
model:

  readout_split(day)  {qubit: (P(1|0), P(0|1))}
  init_error(day)     {qubit: initialization error}, for the qubits IBM reports it for
  days_in_force()     the days on which that is every qubit

Run as a script, it fetches the snapshot in force instead: it asks for the properties at the
day's first job's finish time, as the run did for calibration_after.json, and stops unless
the snapshot's time and readout errors match calibration.json. That needs an IBM Quantum
account (read-only, no QPU time), so nothing runs it by default. IBM no longer serves Fez's
qubit frequencies.

    python docs/analysis/properties.py   # writes results/properties.json
"""

from __future__ import annotations

import json
from datetime import datetime
from functools import cache

from common import RESULTS, RUNS, jobs, read

KEYS = (
    "prob_meas1_prep0",
    "prob_meas0_prep1",
    "readout_error",
    "init_error",
    "T1",
    "T2",
    "readout_length",
)


@cache
def recorded() -> dict:
    return read(RESULTS / "properties.json")


@cache
def readout_in_force(day: str) -> dict[str, float]:
    """The readout error of each qubit in the calibration the day's jobs ran under."""
    return read(RUNS / f"ibm_fez-{day}-r1" / "calibration.json")["readout"]


def in_force(day: str) -> dict[str, dict]:
    """The recorded figures of the qubits whose recorded readout error is the one in force."""
    readout = readout_in_force(day)
    return {q: p for q, p in recorded()[day]["qubits"].items() if p["readout_error"] == readout[q]}


def days_in_force() -> list[str]:
    """The days whose recorded figures are all for the calibration in force (days 1-3)."""
    return [d for d in sorted(recorded()) if len(in_force(d)) == len(recorded()[d]["qubits"])]


def readout_split(day: str) -> dict[int, tuple[float, float]]:
    """(P(1|0), P(0|1)) per qubit, where recorded for the calibration in force."""
    return {
        int(q): (p["prob_meas1_prep0"], p["prob_meas0_prep1"])
        for q, p in in_force(day).items()
        if p["prob_meas1_prep0"] is not None
    }


def init_error(day: str) -> dict[int, float]:
    """IBM's initialization error per qubit, where recorded for the calibration in force."""
    return {
        int(q): p["init_error"] for q, p in in_force(day).items() if p["init_error"] is not None
    }


def fetch() -> dict:
    from qiskit_ibm_runtime import QiskitRuntimeService

    backend = QiskitRuntimeService().backend("ibm_fez")
    out: dict = {}
    for job in jobs():
        if job.day in out:
            continue
        cal = read(job.folder / "calibration.json")
        finished = datetime.fromisoformat(read(job.folder / "job.json")["finished_utc"])
        p = backend.properties(datetime=finished)
        qubits = {}
        for q in range(len(p.qubits)):
            props = p.qubit_property(q)
            qubits[q] = {k: (props[k][0] if k in props else None) for k in KEYS}
        if p.last_update_date != datetime.fromisoformat(cal["calibrated"]) or any(
            v["readout_error"] is not None and v["readout_error"] != cal["readout"][str(q)]
            for q, v in qubits.items()
        ):
            raise RuntimeError(
                f"{job.day}: IBM served {p.last_update_date}, not the calibration in force"
            )
        couplers = {}
        for gate in p.gates:
            if gate.gate == "cz":
                err = next((par.value for par in gate.parameters if par.name == "gate_error"), None)
                couplers["-".join(map(str, sorted(gate.qubits)))] = err
        out[job.day] = {
            "calibrated": cal["calibrated"],
            "served": str(p.last_update_date),
            "qubits": qubits,
            "cz": couplers,
        }
        print(job.day, "served", p.last_update_date, flush=True)
    return out


def main() -> None:
    (RESULTS / "properties.json").write_text(json.dumps(fetch(), indent=1) + "\n")


if __name__ == "__main__":
    main()
