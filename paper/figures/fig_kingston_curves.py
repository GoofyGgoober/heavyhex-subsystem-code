"""Kingston depth-level data and the simulated curves' exponential residuals."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from make_figures import D3, D5, GRID, INK, MUTED, PAGE, load


def make(results, output):
    curves = load(results / "kingston_checks.json")["curves"]
    n = np.arange(1, 5)
    fig, axes = plt.subplots(1, 4, figsize=(PAGE, 1.35), sharex=True, layout="constrained")
    for col, key in enumerate(("X3", "X5", "Z3", "Z5")):
        rows, ax = curves[key], axes[col]
        for variant, color in (("ibm", D5), ("measured_tq", D3)):
            row = rows[variant]
            ax.plot(n, 100 * np.array(row["residual_P"]), "o-", color=color, ms=2, lw=0.6)
        ax.set_title(f"{key[0]} memory, $d={key[1]}$", loc="left", color=INK, fontsize=7)
        ax.axhline(0, color=MUTED, lw=0.6)
        ax.set(xlabel="rounds", xticks=n, ylim=(-0.55, 0.55))
        ax.grid(axis="y", color=GRID, lw=0.4)
    axes[0].set_ylabel("model - fit\n(points)")
    _legend(fig, device=False)
    fig.savefig(output)
    plt.close(fig)


def make_main(results, output):
    curves = load(results / "kingston_checks.json")["curves"]
    n = np.arange(1, 5)
    fig, axes = plt.subplots(
        1, 4, figsize=(PAGE, 1.7), sharex=True, sharey=True, layout="constrained"
    )
    for ax, key in zip(axes, ("X3", "X5", "Z3", "Z5")):
        rows = curves[key]
        observed = rows["observed"]
        p = np.array(observed["P"])
        se = np.sqrt(p * (1 - p) / observed["shots"])
        ax.errorbar(n, p, yerr=se, fmt="o", color=INK, ms=2.5, elinewidth=0.6)
        for variant, color in (("ibm", D5), ("measured_tq", D3)):
            ax.plot(n, rows[variant]["P"], "s-", color=color, ms=2, lw=0.7)
        ax.set_title(f"{key[0]} memory, $d={key[1]}$", loc="left", color=INK, fontsize=7)
        ax.set(ylim=(0, 0.45), yticks=[0, 0.2, 0.4], xticks=n, xlabel="rounds")
        ax.grid(axis="y", color=GRID, lw=0.4)
    axes[0].set_ylabel(r"$P_\mathrm{L}(n)$")
    _legend(fig, device=True)
    fig.savefig(output)
    plt.close(fig)


def _legend(fig, *, device):
    fig.legend(
        handles=(
            [
                Line2D([], [], marker="o", color=INK, ls="none", ms=3, label="device"),
            ]
            if device
            else []
        )
        + [
            Line2D([], [], marker="s", color=D5, lw=0.7, ms=3, label="IBM inputs"),
            Line2D([], [], marker="s", color=D3, lw=0.7, ms=3, label="measured $T_q$"),
        ],
        loc="outside upper center",
        ncol=4,
        fontsize=6.5,
    )
