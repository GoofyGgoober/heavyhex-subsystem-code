"""Daily matched-decoder discrepancies and the observed persistent flag records."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from make_figures import D3, D5, GRID, INK, MUTED, PAGE, load


def make(results, output):
    primary = load(results / "stored_priors.json")["primary"]
    flags = load(results / "flags_summary.json")["refiring"]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.15), layout="constrained")
    ax = axes[0]
    for basis, dash in (("X", "-"), ("Z", "--")):
        for d, color, marker in (("3", D3, "s"), ("5", D5, "o")):
            values = np.exp(primary[f"{basis} eps{d}"]["day_log_means"])
            ax.plot(
                range(1, 6),
                values,
                color=color,
                marker=marker,
                ms=3,
                ls=dash,
                lw=0.8,
                label=f"{basis}, $d={d}$",
            )
    ax.axhline(1, color=MUTED, lw=0.7)
    ax.set(xlabel="day", ylabel="device / stored model", xticks=range(1, 6), ylim=(0.9, 3.1))
    ax.set_title("(a) Daily geometric means", loc="left", color=INK)
    ax.legend(ncol=2, fontsize=6.5)
    ax.grid(axis="y", color=GRID, lw=0.4)

    ax = axes[1]
    for basis, color, marker in (("X", D3, "s"), ("Z", D5, "o")):
        row = flags[basis]
        lag = np.array(row["lags"])
        ax.plot(
            lag, 100 * np.array(row["device"]), marker, color=color, ms=3, label=f"{basis} records"
        )
        ax.plot(lag, 100 * np.array(row["hmm_implied"]["curve"]), color=color, lw=0.9)
        model = row["model_lags_1_6"]
        ax.plot(lag[: len(model)], 100 * np.array(model), color=MUTED, ls="--", lw=0.7)
    ax.axhline(0, color=MUTED, lw=0.4)
    ax.set(
        xlabel="lag (rounds)",
        ylabel=r"persistence $\Delta$ (percentage points)",
        xticks=range(1, 8),
        ylim=(-0.6, 17),
    )
    ax.set_title("(b) Flag and relay persistence", loc="left", color=INK)
    ax.legend(fontsize=6.5)
    ax.grid(axis="y", color=GRID, lw=0.4)
    fig.savefig(output)
    plt.close(fig)
