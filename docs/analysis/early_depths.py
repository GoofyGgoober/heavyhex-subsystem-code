"""Fixed-depth Fez forecast errors, with days as the uncertainty blocks.

Device and forecast both use f=1 priors. Average the two placement differences
within each day, then the five days. The primary family comprises sixteen cells:
two bases, two distances, four depths. Its Bonferroni t4 intervals, set after the run,
cover these cells only. Also the range of exact ML's change to each depth's logical
error, device and model (decoder_ml.json, n = 1-4).

    python docs/analysis/early_depths.py
"""

from __future__ import annotations

import numpy as np
from scipy.stats import t

from common import day_number, result, write_json


def interval(daily: np.ndarray) -> dict:
    daily = np.asarray(daily, float)
    if daily.shape != (5,) or not np.isfinite(daily).all():
        raise ValueError("five finite daily differences are required")
    se = daily.std(ddof=1) / np.sqrt(5)
    mean = float(daily.mean())
    half = t.ppf(1 - 0.05 / (2 * 16), 4) * se
    omissions = {}
    for day in range(5):
        subset = np.delete(daily, day)
        center = float(subset.mean())
        width = t.ppf(1 - 0.05 / (2 * 16), 3) * subset.std(ddof=1) / np.sqrt(4)
        omissions[str(day + 1)] = {
            "mean": center,
            "ci95_family16": [center - width, center + width],
        }
    return {
        "mean": mean,
        "daily": daily.tolist(),
        "ci95_family16": [mean - half, mean + half],
        "leave_one_day_out": omissions,
    }


def main() -> None:
    matched = result("stored_priors.json")["curves"]
    stored = result("fits_counts.json")["jobs"]
    reps = result("fits.json")["per_rep"]
    cells = {}
    for basis in "XZ":
        for d in "35":
            cells[basis + d] = {}
            for depth in range(1, 5):
                by_day = {}
                for rep in reps.values():
                    job = rep["jobs"][d]
                    observed = matched[job][basis][d]["P"][depth - 1]
                    predicted = stored[job]["predicted"]["frozen_f1.0"]["P"][basis][d][depth - 1]
                    by_day.setdefault(day_number(rep["day"]), []).append(observed - predicted)
                if sorted(by_day) != [1, 2, 3, 4, 5] or any(len(v) != 2 for v in by_day.values()):
                    raise ValueError("every day must contain both placements")
                daily = np.array([np.mean(by_day[day]) for day in sorted(by_day)])
                cells[basis + d][str(depth)] = interval(daily)
    ml = result("decoder_ml.json")
    device, model, gap = [], [], []
    for patch in ml["patches"].values():
        for row in patch["by_rounds"]:
            device.append(row["device_ml"] - row["device_matching"])
            model.append(row["model_ml"] - row["model_matching"])
            gap.append(row["device_ml"] - row["model_ml"])
    out = {
        "what": "Primary fixed-depth comparison; paired placements averaged within each day; "
        "five-day mean probability differences and approximate simultaneous 95% t4 intervals "
        "over 16 cells. Leave-one-day-out uses t3 on four days, a precision sensitivity "
        "diagnostic, not independent tests. The forecast, inputs and placements are those stored before "
        "the run; the intervals are not prediction intervals for new runs.",
        "cells": cells,
        "exact_ml_depth_diagnostic": {
            "window": [1, 2, 3, 4],
            "device_probability_change_range": [min(device), max(device)],
            "model_probability_change_range": [min(model), max(model)],
            "device_minus_model_after_ml_range": [min(gap), max(gap)],
            "note": "Uses the stored failure probabilities at each depth, not log-slope rates",
        },
    }
    write_json("early_depths.json", out)
    for key, depths in cells.items():
        print(
            key,
            [
                (
                    n,
                    round(100 * row["mean"], 3),
                    np.round(100 * np.array(row["ci95_family16"]), 3).tolist(),
                )
                for n, row in depths.items()
            ],
        )


if __name__ == "__main__":
    main()
