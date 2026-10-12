"""Figure for the heating test: pooled, normalized error rates in each tenth of a job.

    python docs/diagnosis/revision/heating/figure.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
R = json.loads((HERE / "heating.json").read_text())
SECONDS = float(np.mean(list(R["seconds_per_job_of_shots"].values())))
CHIP, MODEL, INK, MUTED, GRID = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#e4e3df"

PANELS = [
    ("detection_events", "Stabilizer detection events", True),
    ("flags_relays_Z_memory", "Flag and relay firings (Z memory)", True),
    ("logical_failure", "Decoded logical failure (per circuit)", True),
    ("logical_eps_per_round", "Error per round, fitted per bin", True),
    ("idle_ancilla_ones", "Idle test: ancilla readouts = 1", False),
    ("idle_contrast_loss", "Idle test: data contrast loss", False),
]


def main() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.8), sharex=True)
    for ax, (key, title, has_null) in zip(axes.flat, PANELS):
        r = R["time_H1"][key]
        t = np.array(r["pooled_bins"]["t"]) * SECONDS
        y, e = np.array(r["pooled_bins"]["y"]), np.array(r["pooled_bins"]["se"])
        if has_null:
            n = R["null_model_shots"][key]["pooled_bins"]
            ax.errorbar(t + 0.12, n["y"], yerr=n["se"], fmt="o", ms=4, color=MODEL, lw=1.2,
                        capsize=0, label="model shots (no drift)")
        ax.errorbar(t, y, yerr=e, fmt="o", ms=5, color=CHIP, lw=1.6, capsize=0, label="chip")
        re = r["across_jobs"]["random"]
        s, lo, hi = 100 * re["slope"], 100 * re["ci95"][0], 100 * re["ci95"][1]
        ax.axhline(1, color=GRID, lw=1, zorder=0)
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=18)
        ax.text(0, 1.02, f"start→end {s:+.1f}%, 95% CI [{lo:+.1f}, {hi:+.1f}]; "
                f"up in {r['across_jobs']['positive_slopes']}/{r['across_jobs']['jobs']} jobs",
                transform=ax.transAxes, fontsize=8, color=MUTED, va="bottom")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8.5)
        ax.grid(axis="y", color=GRID, lw=0.6)
    for ax in axes[1]:
        ax.set_xlabel("time into the job's shots (s)", fontsize=9, color=MUTED)
    for ax in axes[:, 0]:
        ax.set_ylabel("rate / its job mean", fontsize=9, color=MUTED)
    axes[0, 0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("Error rates over each job, pooled over 15 jobs (10 time bins; ±1 s.e.)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(HERE / "heating.png", dpi=150, facecolor="white")


if __name__ == "__main__":
    main()
