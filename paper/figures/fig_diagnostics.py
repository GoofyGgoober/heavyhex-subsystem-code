"""Day-resolved distance ratios and the observed persistent flag records."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from make_figures import D3, D5, GRID, INK, MUTED, PAGE, load


def make(results, output):
    fits = load(results / "fits.json")
    flags = load(results / "flags_summary.json")["refiring"]
    fig, axes = plt.subplots(1, 2, figsize=(PAGE, 2.15), layout="constrained")
    ax = axes[0]
    reps = sorted(
        (r for r in fits["per_rep"].values() if r["placement"] == 1), key=lambda r: r["day"]
    )
    observed, low, high, predicted = [], [], [], []
    for rep in reps:
        row = rep["X"]["lambda"]
        observed.append(row["binomial"])
        low.append(row["binomial_lo68_quasi"])
        high.append(row["binomial_hi68_quasi"])
        model = fits["predictions"][rep["jobs"]["3"]]["predicted_at_measured_dephasing"]["X"]
        predicted.append(model["3"]["binomial"]["eps"] / model["5"]["binomial"]["eps"])
    observed = np.array(observed)
    ax.errorbar(
        range(1, 6),
        observed,
        yerr=[observed - low, high - observed],
        fmt="o-",
        color=D5,
        ms=3,
        lw=0.8,
        label="device",
    )
    ax.plot(range(1, 6), predicted, "s--", color=D3, ms=3, lw=0.8, label="in-job model")
    ax.annotate(
        "day 2",
        xy=(2, predicted[1]),
        xytext=(2.35, 0.87),
        fontsize=6.5,
        arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.5},
    )
    ax.set(xlabel="day", ylabel=r"$\Lambda_\mathrm{X}$", xticks=range(1, 6), ylim=(0.45, 0.92))
    ax.set_title("(a) Distance ratio, pair 1", loc="left", color=INK)
    ax.legend(loc="lower right", fontsize=6.5)
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
