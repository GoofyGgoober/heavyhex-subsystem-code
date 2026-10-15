"""Primary fixed-depth forecast errors, with paired placements blocked by day."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from make_figures import D3, D5, GRID, INK, MUTED, PAGE, load


def make(results, output):
    cells = load(results / "early_depths.json")["cells"]
    fig, axes = plt.subplots(2, 2, figsize=(PAGE, 2.8), sharex=True, layout="constrained")
    n = np.arange(1, 5)
    for ax, key in zip(axes.flat, ("X3", "X5", "Z3", "Z5")):
        rows = [cells[key][str(depth)] for depth in n]
        daily = 100 * np.array([r["daily"] for r in rows]).T
        for values in daily:
            ax.plot(n, values, color=GRID, lw=0.6, zorder=1)
        mean = 100 * np.array([r["mean"] for r in rows])
        ci = 100 * np.array([r["ci95_family16"] for r in rows])
        color = D3 if key[1] == "3" else D5
        ax.errorbar(
            n,
            mean,
            yerr=[mean - ci[:, 0], ci[:, 1] - mean],
            fmt="o",
            color=color,
            ms=3,
            lw=0.8,
            capsize=2,
            zorder=3,
        )
        ax.axhline(0, color=MUTED, lw=0.7)
        ax.set_title(f"{key[0]} memory, $d={key[1]}$", loc="left", color=INK, fontsize=8)
        ax.set(xticks=n, ylabel="device - model (points)")
        ax.grid(axis="y", color=GRID, lw=0.4)
    for ax in axes[1]:
        ax.set_xlabel("rounds")
    fig.savefig(output)
    plt.close(fig)
