"""Primary fixed-depth forecast errors, with paired placements blocked by day."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from make_figures import D3, D5, GRID, INK, MUTED, PAGE, ROUNDS, load, p_of_n


def make(results, output):
    cells = load(results / "early_depths.json")["cells"]
    fig, axes = plt.subplots(2, 4, figsize=(PAGE, 3.1), layout="constrained")
    n = np.arange(1, 5)
    for col, key in enumerate(("X3", "X5", "Z3", "Z5")):
        color = D3 if key[1] == "3" else D5
        ax = axes[0, col]
        for source, marker, line, ink in (
            ("observed", "o", "none", INK),
            ("stored", "s", "-", color),
        ):
            values = np.mean(
                [p_of_n(rep, int(key[1]), key[0], source) for rep in (1, 2)], axis=(0, 1)
            )
            ax.plot(ROUNDS, values, marker=marker, ls=line, color=ink, ms=2.6, lw=0.8)
        ax.axhline(0.5, color=MUTED, ls="--", lw=0.6)
        ax.set(xticks=[1, 2, 4, 6, 8], ylim=(0, 0.55), xlabel="rounds")
        ax.set_title(f"{key[0]} memory, $d={key[1]}$", loc="left", color=INK, fontsize=8)
        ax.grid(axis="y", color=GRID, lw=0.4)
        ax = axes[1, col]
        rows = [cells[key][str(depth)] for depth in n]
        daily = 100 * np.array([r["daily"] for r in rows]).T
        for values in daily:
            ax.plot(n, values, color=GRID, lw=0.6, zorder=1)
        mean = 100 * np.array([r["mean"] for r in rows])
        ci = 100 * np.array([r["ci95_family16"] for r in rows])
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
        ax.set(xticks=n, xlabel="rounds", ylim=(-1, 13) if key[0] == "X" else (-4, 25))
        ax.grid(axis="y", color=GRID, lw=0.4)
    axes[0, 0].set_ylabel(r"$P_\mathrm{L}(n)$")
    axes[1, 0].set_ylabel("device - model\n(points)")
    fig.legend(
        handles=[
            Line2D([], [], marker="o", color=INK, ls="none", ms=3, label="device mean"),
            Line2D([], [], marker="s", color=MUTED, lw=0.8, ms=3, label="stored-model mean"),
            Line2D([], [], color=MUTED, lw=0.6, ls="--", label=r"$P_\mathrm{L}=1/2$"),
        ],
        loc="outside upper center",
        ncol=3,
        fontsize=7,
    )
    fig.savefig(output)
    plt.close(fig)
