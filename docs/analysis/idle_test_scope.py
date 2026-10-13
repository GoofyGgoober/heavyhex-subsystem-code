"""What the idle test sees of the memory circuits' idle noise (Sec. pred, App. f).

* How often the ancilla, flag and relay readouts read 1, per job, in the idle circuits and in
  the memory circuits. In the idle circuits every CX is a wait, so the ancillas stay near |0>
  and the conditional resets rarely act; in memory they read 1 about a third of the time.
* The reported T2 of the qubit-jobs that the idle test does not pin to +-0.5, which are left
  out of the job's median f (idle_f_t2.json), against the qubit-jobs it uses.

Seconds.

    python docs/analysis/idle_test_scope.py      # writes results/idle_test_scope.json
"""

from __future__ import annotations

import statistics

import numpy as np

from heavyhex.experiment.circuits import SETTINGS

from common import jobs, result, write_json


def ones(job) -> dict[str, float]:
    """Mean fraction of 1s over the non-data readouts, idle and memory circuits apart."""
    found: dict[str, list[float]] = {"idle": [], "memory": []}
    for setting, bits in zip(SETTINGS, job.shots):
        for name, b in bits.items():
            if not name.startswith("d"):  # m3, m5: every ancilla, flag and relay readout
                found[setting.kind].append(float(b.mean()))
    return {kind: float(np.mean(v)) for kind, v in found.items()}


def main() -> None:
    per_job = {job.name: ones(job) for job in jobs()}
    t2 = result("idle_f_t2.json")["f_vs_t2"]
    left_out = [q["t2_us"] for q in t2["excluded_measured_false"]]
    used = [q["t2_us"] for q in t2["per_qubit_job"]]
    out = {
        "ancilla_reads_1": {
            "per_job": per_job,
            "idle_range": [
                min(v["idle"] for v in per_job.values()),
                max(v["idle"] for v in per_job.values()),
            ],
            "memory_range": [
                min(v["memory"] for v in per_job.values()),
                max(v["memory"] for v in per_job.values()),
            ],
        },
        "f_left_out_of_the_median": {
            "qubit_jobs": len(left_out),
            "median_t2_us": statistics.median(left_out),
            "t2_us": sorted(left_out),
            "used_qubit_jobs": len(used),
            "used_median_t2_us": statistics.median(used),
        },
    }
    write_json("idle_test_scope.json", out)
    a, f = out["ancilla_reads_1"], out["f_left_out_of_the_median"]
    print(
        f"ancillas read 1: idle {a['idle_range'][0]:.1%}-{a['idle_range'][1]:.1%},",
        f"memory {a['memory_range'][0]:.1%}-{a['memory_range'][1]:.1%};",
        f"left out of the median: {f['qubit_jobs']} qubit-jobs, median T2 {f['median_t2_us']:.0f} us",
        f"against {f['used_median_t2_us']:.0f} us",
    )


if __name__ == "__main__":
    main()
