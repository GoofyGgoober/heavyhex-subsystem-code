"""Do flags and relays remember? The first sign of leakage, with no model of it.

In 8-round memory every flag and relay should read 0 each round, in either basis (they are
uncomputed), and in the calibration model the rounds are independent. For each readout series,
P(fires at r + L | fired at r) minus P(fires at r + L | quiet at r), averaged over series, on the
chip and on the model's own shots; and how many shots hold a run of 5 or more firings in a row.
The base odds behind the excess are written too: the fraction of rounds in which a flag
or relay fires, and P(fires at r + L | quiet at r) and P(fires at r + L | fired at r)
themselves, each averaged over series in the same way. Chip numbers repeat exactly; the
model's come from stim-sampled shots (seed 6), which repeat only with the same stim build.
Z memory is the paper's Fig. leak (a); X memory (common.readout_model) is its replication.
A few minutes.

    python docs/analysis/refiring.py              # Z memory: results/refiring.json
    python docs/analysis/refiring.py --basis X    # X memory: results/refiring_x.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from common import RESULTS, ROUND_US, Job, job_folders, memory_index, readout_model, readout_series

LAGS = range(1, 7)


def excess(x: np.ndarray, lag: int) -> float:
    now, later = x[:, :-lag], x[:, lag:]
    return float(
        (now & later).sum() / max(now.sum(), 1) - (~now & later).sum() / max((~now).sum(), 1)
    )


def odds(x: np.ndarray, lag: int) -> tuple[float, float]:
    """(P(fires at r + lag | quiet at r), P(fires at r + lag | fired at r)), as in excess()."""
    now, later = x[:, :-lag], x[:, lag:]
    return (
        float((~now & later).sum() / max((~now).sum(), 1)),
        float((now & later).sum() / max(now.sum(), 1)),
    )


def longest(x: np.ndarray) -> np.ndarray:
    best = np.zeros(len(x), int)
    cur = np.zeros(len(x), int)
    for r in range(x.shape[1]):
        cur = np.where(x[:, r], cur + 1, 0)
        best = np.maximum(best, cur)
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    ap.add_argument("--basis", choices=("Z", "X"), default="Z", help="memory basis (default Z)")
    basis = ap.parse_args().basis
    idx = memory_index(basis, 8)
    rows = {src: {lag: [] for lag in LAGS} for src in ("chip", "model")}
    runs = {"chip": [], "model": []}
    positive = []
    rate = {"chip": [], "model": []}
    cond = {
        src: {lag: {"after_quiet": [], "after_firing": []} for lag in LAGS}
        for src in ("chip", "model")
    }
    for job in map(Job, job_folders()):  # one job in memory at a time
        model, records, labels = readout_model(job, idx)
        chip, _ = job.chip_detectors(idx, model, records)
        sim = np.asarray(model.compile_detector_sampler(seed=6).sample(20000))
        for cols in readout_series(labels).values():
            for lag in LAGS:
                rows["chip"][lag].append(excess(chip[:, cols], lag))
                rows["model"][lag].append(excess(sim[:, cols], lag))
                for src, x in (("chip", chip[:, cols]), ("model", sim[:, cols])):
                    quiet, fired = odds(x, lag)
                    cond[src][lag]["after_quiet"].append(quiet)
                    cond[src][lag]["after_firing"].append(fired)
            for src, x in (("chip", chip[:, cols]), ("model", sim[:, cols])):
                rate[src].append(float(x.mean()))
            positive.append(rows["chip"][1][-1] > rows["model"][1][-1])
            runs["chip"].append(float((longest(chip[:, cols]) >= 5).mean()))
            runs["model"].append(float((longest(sim[:, cols]) >= 5).mean()))
    out = {
        "basis": basis,
        "lags": {lag: {src: float(np.mean(rows[src][lag])) for src in rows} for lag in LAGS},
        "series": len(positive),
        "chip_above_model_at_lag_1": float(np.mean(positive)),
        "runs_of_5": {src: float(np.mean(v)) for src, v in runs.items()},
    }
    # Base odds, averaged over series: the share of rounds that fire, and the two conditional
    # probabilities whose difference is the excess above (lag 1 in "base_odds", every lag below).
    out["base_odds"] = {
        src: {
            "fires_per_round": float(np.mean(rate[src])),
            "after_quiet": float(np.mean(cond[src][1]["after_quiet"])),
            "after_firing": float(np.mean(cond[src][1]["after_firing"])),
        }
        for src in rate
    }
    out["conditional_by_lag"] = {
        lag: {src: {k: float(np.mean(v)) for k, v in cond[src][lag].items()} for src in cond}
        for lag in LAGS
    }
    for lag in LAGS:
        for src in cond:
            c = out["conditional_by_lag"][lag][src]
            assert abs(c["after_firing"] - c["after_quiet"] - out["lags"][lag][src]) < 1e-12
    for lag in LAGS:
        print(
            f"lag {lag} ({lag * ROUND_US:.0f} us): chip {out['lags'][lag]['chip']:+.3f}  model {out['lags'][lag]['model']:+.3f}"
        )
    print(
        f"{out['series']} series; chip above model at lag 1 in {100 * out['chip_above_model_at_lag_1']:.0f}%; "
        f"a run of 5+ firings: chip {100 * out['runs_of_5']['chip']:.2f}% of shots per series, model {100 * out['runs_of_5']['model']:.3f}%"
    )
    for src, b in out["base_odds"].items():
        print(
            f"{src}: fires in {100 * b['fires_per_round']:.1f}% of rounds; next round {100 * b['after_quiet']:.1f}% after a quiet round, "
            f"{100 * b['after_firing']:.1f}% after a firing (difference {100 * (b['after_firing'] - b['after_quiet']):.1f} points)"
        )
    for lag in LAGS:
        c = out["conditional_by_lag"][lag]["chip"]
        print(
            f"chip, lag {lag} ({lag * ROUND_US:.1f} us): {100 * c['after_quiet']:.1f}% after quiet, "
            f"{100 * c['after_firing']:.1f}% after firing, excess {100 * out['lags'][lag]['chip']:.1f} points"
        )
    name = "refiring.json" if basis == "Z" else "refiring_x.json"
    (RESULTS / name).write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
