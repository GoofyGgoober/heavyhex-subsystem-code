"""How precisely IBM's reported readout errors pin the model's inputs (Sec. model).

Every reported readout error in runs/*/calibration.json is a whole number of 1/8192, and every
P(1|0) and P(0|1) in results/properties.json a whole number of 1/4096: each comes from 4096
shots per prepared state. The binomial standard error of e = (P(1|0) + P(0|1)) / 2 is then

    sigma_e = sqrt(P(1|0) (1 - P(1|0)) + P(0|1) (1 - P(0|1))) / (2 sqrt(4096)),

at most sqrt(e (1 - e) / 8192), the value with the two equal. The split is used where it is
recorded for the calibration in force (properties.readout_split: days 1-3) and its mean is the
reported e; elsewhere the bound stands in. On a few qubits a day the reported e is not the
mean of the split IBM reports beside it; they are counted. Over the qubits the jobs use (every
qubit of each job's patches), as sigma_e / e.

A second.

    python docs/analysis/input_resolution.py     # writes results/input_resolution.json
"""

from __future__ import annotations

from math import sqrt

import numpy as np

from common import jobs, write_json
from properties import readout_split

SHOTS_PER_STATE = 4096


def main() -> None:
    whole, values, rows, mismatched = 0, 0, [], set()
    for job in jobs():
        readout = job.calibration.readout
        counts = np.array(list(readout.values())) * 2 * SHOTS_PER_STATE
        whole += int(np.isclose(counts, np.round(counts), rtol=0, atol=1e-6).sum())
        values += len(counts)
        split = readout_split(job.day)
        mismatched |= {
            (job.day, q) for q, (p10, p01) in split.items() if (p10 + p01) / 2 != readout[q]
        }
        used = sorted({q for chip in job.info["chips"].values() for q in chip})
        for q in used:
            e = readout[q]
            exact = q in split and (job.day, q) not in mismatched
            if exact:
                p10, p01 = split[q]
                sigma = sqrt(p10 * (1 - p10) + p01 * (1 - p01)) / (2 * sqrt(SHOTS_PER_STATE))
            else:
                sigma = sqrt(e * (1 - e) / (2 * SHOTS_PER_STATE))
            rows.append({"job": job.name, "qubit": q, "e": e, "sigma": sigma, "split": exact})
    e = np.array([r["e"] for r in rows])
    rel = np.array([r["sigma"] / r["e"] for r in rows])
    days_with_split = sorted({d for d, _ in mismatched})
    out = {
        "what": "binomial standard error of IBM's reported readout error (4096 shots per prepared "
        "state), relative to the error, over the qubits of every job's patches",
        "readout_values_whole_multiples_of_1_over_8192": [whole, values],
        "qubit_jobs": len(rows),
        "with_split": sum(r["split"] for r in rows),
        "reported_e_not_the_mean_of_the_split (qubits per day)": {
            d: sorted(q for dd, q in mismatched if dd == d) for d in days_with_split
        },
        "e_median": float(np.median(e)),
        "e_quartiles": np.percentile(e, [25, 75]).tolist(),
        "relative_sigma_median": float(np.median(rel)),
        "relative_sigma_quartiles": np.percentile(rel, [25, 75]).tolist(),
        "relative_sigma_range": [float(rel.min()), float(rel.max())],
        "relative_sigma_at_1_percent": sqrt(0.99 / (0.01 * 2 * SHOTS_PER_STATE)),
        "step_1_over_8192_relative_to_median_e": 1 / (2 * SHOTS_PER_STATE) / float(np.median(e)),
        "qubit_jobs_below_0.5_percent": int((e < 0.005).sum()),
    }
    write_json("input_resolution.json", out)
    q1, q3 = out["relative_sigma_quartiles"]
    print(
        f"{whole}/{values} readout values are whole multiples of 1/8192; {len(rows)} qubit-jobs "
        f"used, {out['with_split']} with the split; reported e is not the split's mean on "
        f"{len(mismatched)} qubit-days. Readout error median {100 * out['e_median']:.2f}%; "
        f"sigma_e / e median {100 * out['relative_sigma_median']:.1f}% (quartiles "
        f"{100 * q1:.1f}-{100 * q3:.1f}%, range {100 * rel.min():.1f}-{100 * rel.max():.1f}%); "
        f"at e = 1%: {100 * out['relative_sigma_at_1_percent']:.1f}%"
    )


if __name__ == "__main__":
    main()
