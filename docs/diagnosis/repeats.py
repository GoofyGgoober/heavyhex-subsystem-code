"""Do flags and relays remember? The first sign of leakage, with no model of it.

In 8-round Z memory every flag and relay should read 0 each round, and in the calibration
model the rounds are independent. For each readout series, P(fires at r + L | fired at r)
minus P(fires at r + L | quiet at r), averaged over series, on the chip and on the
model's own shots; and how many shots hold a run of 5 or more firings in a row.

    python docs/diagnosis/repeats.py
"""

from __future__ import annotations

import json

import numpy as np

from common import REPO, ROUND_US, jobs, memory_index, readout_series

LAGS = range(1, 7)


def excess(x: np.ndarray, lag: int) -> float:
    now, later = x[:, :-lag], x[:, lag:]
    return float((now & later).sum() / max(now.sum(), 1) - (~now & later).sum() / max((~now).sum(), 1))


def longest(x: np.ndarray) -> np.ndarray:
    best = np.zeros(len(x), int); cur = np.zeros(len(x), int)
    for r in range(x.shape[1]):
        cur = np.where(x[:, r], cur + 1, 0); best = np.maximum(best, cur)
    return best


def main() -> None:
    idx = memory_index("Z", 8)
    rows = {src: {lag: [] for lag in LAGS} for src in ("chip", "model")}
    runs = {"chip": [], "model": []}
    positive = []
    for job in jobs():
        model, records, labels = job.model(idx)
        chip, _ = job.chip_detectors(idx, model, records)
        sim = model.compile_detector_sampler(seed=6).sample(20000)
        for cols in readout_series(labels).values():
            for lag in LAGS:
                rows["chip"][lag].append(excess(chip[:, cols], lag))
                rows["model"][lag].append(excess(sim[:, cols], lag))
            positive.append(rows["chip"][1][-1] > rows["model"][1][-1])
            runs["chip"].append(float((longest(chip[:, cols]) >= 5).mean()))
            runs["model"].append(float((longest(sim[:, cols]) >= 5).mean()))
    out = {"lags": {lag: {src: float(np.mean(rows[src][lag])) for src in rows} for lag in LAGS},
           "series": len(positive), "chip_above_model_at_lag_1": float(np.mean(positive)),
           "runs_of_5": {src: float(np.mean(v)) for src, v in runs.items()}}
    for lag in LAGS:
        print(f"lag {lag} ({lag * ROUND_US:.0f} us): chip {out['lags'][lag]['chip']:+.3f}  model {out['lags'][lag]['model']:+.3f}")
    print(f"{out['series']} series; chip above model at lag 1 in {100*out['chip_above_model_at_lag_1']:.0f}%; "
          f"a run of 5+ firings: chip {100*out['runs_of_5']['chip']:.2f}% of shots per series, model {100*out['runs_of_5']['model']:.3f}%")
    (REPO / "docs/diagnosis/results/repeats.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
