"""Leave-one-day-out sensitivity of the matched stored-prior Fez comparison.

The five daily means are already saved by stored_priors.py. Each omission recomputes
the same family-of-five t interval on the four remaining days. These overlapping
subsets are diagnostics, not five new confirmatory tests or independent experiments.

    python docs/analysis/day_sensitivity.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t

from common import result, write_json
from fitting import mle_batch


def leave_one_out(daily) -> dict:
    y = np.asarray(daily, float)
    if y.shape != (5,) or not np.isfinite(y).all():
        raise ValueError("Five finite daily log means are required")
    out = {}
    for day in range(5):
        subset = np.delete(y, day)
        half = t.ppf(0.995, 3) * subset.std(ddof=1) / np.sqrt(4)
        out[str(day + 1)] = {
            "ratio": float(np.exp(subset.mean())),
            "ci95_family_of_five": np.exp([subset.mean() - half, subset.mean() + half]).tolist(),
        }
    return out


def main() -> None:
    source = result("stored_priors.json")
    out = {
        "what": "Matched stored-prior comparison, omit each day in turn; four remaining "
        "daily log means, t with three degrees of freedom, Bonferroni family of five. "
        "Selected-subset sensitivity only, not confirmatory tests.",
        "by_quantity": {
            q: leave_one_out(row["day_log_means"]) for q, row in source["primary"].items()
        },
        "window_boundaries": window_boundaries(source),
    }
    write_json("day_sensitivity.json", out)
    for q, rows in out["by_quantity"].items():
        crossings = [day for day, row in rows.items() if row["ci95_family_of_five"][0] <= 1]
        print(q, "interval includes one after omitting days:", crossings)


def window_boundaries(source: dict) -> dict:
    """Check the retained P(1)>=0 constraint at each best fit, not its profile interval."""
    from stored_priors import WINDOWS

    original = result("fits_counts.json")["jobs"]
    hardware, simulated, labels = [], [], []
    for job, bases in source["curves"].items():
        for basis, patches in bases.items():
            for patch, curve in patches.items():
                hardware.append(curve["k"])
                labels.append([job, basis, patch])
                pred = original[job]["predicted"]["frozen_f1.0"]
                simulated.append(np.round(np.array(pred["P"][basis][patch]) * pred["shots"]))
    out = {}
    for name, counts, shots in (("device", hardware, 2500), ("model", simulated, 200000)):
        out[name] = {}
        for window, ns in WINDOWS.items():
            ix = [[1, 2, 3, 4, 6, 8].index(n) for n in ns]
            eps, amplitude, _ = mle_batch(np.array(counts)[:, ix], shots, np.array(ns))
            p1 = (1 - amplitude * (1 - 2 * eps)) / 2
            out[name][window] = {
                "active_at_mle": [labels[i] for i in np.flatnonzero(p1 < 1e-6)],
                "minimum_fitted_P1": float(p1.min()),
            }
    return out


if __name__ == "__main__":
    main()
