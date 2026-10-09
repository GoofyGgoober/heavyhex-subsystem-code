"""IBM's full per-qubit properties for each run day's prepared calibration (read-only).

The run folders keep only what the model uses (CZ, readout, T1, T2, single-qubit error).
For the diagnosis this adds each qubit's two readout errors P(1|0) and P(0|1) and its
reset ("init") error, and each coupler's CZ error, as IBM served them at the time of the
calibration each day's jobs were prepared from (IBM serves the latest snapshot at or
before that time). IBM no longer serves Fez's qubit frequencies.

    python docs/diagnosis/fetch_properties.py   # writes docs/diagnosis/results/properties.json
"""

from __future__ import annotations

import json
from datetime import datetime

from qiskit_ibm_runtime import QiskitRuntimeService

from common import REPO, jobs


def main() -> None:
    backend = QiskitRuntimeService().backend("ibm_fez")
    out: dict = {}
    for job in jobs():
        stamp = json.loads((job.folder / "calibration.json").read_text())["calibrated"]
        if job.day in out:
            continue
        p = backend.properties(datetime=datetime.fromisoformat(stamp))
        qubits = {}
        for q in range(len(p.qubits)):
            props = p.qubit_property(q)
            qubits[q] = {k: (props[k][0] if k in props else None) for k in
                         ("prob_meas1_prep0", "prob_meas0_prep1", "readout_error", "init_error",
                          "T1", "T2", "readout_length")}
        couplers = {}
        for gate in p.gates:
            if gate.gate == "cz":
                err = next((par.value for par in gate.parameters if par.name == "gate_error"), None)
                couplers["-".join(map(str, sorted(gate.qubits)))] = err
        out[job.day] = {"calibrated": stamp, "served": str(p.last_update_date), "qubits": qubits, "cz": couplers}
        print(job.day, "served", p.last_update_date, flush=True)
    path = REPO / "docs/diagnosis/results/properties.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
