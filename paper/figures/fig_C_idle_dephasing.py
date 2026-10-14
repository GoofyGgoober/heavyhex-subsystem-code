"""App. f figure: the idle-test decay against its references, and f against T2.

Draws from docs/analysis/results/idle_f_t2.json (docs/analysis/idle_f_t2.py) in the style of
make_figures.py, whose fonts and rcParams it imports. Run from the repository root:

    python paper/figures/fig_C_idle_dephasing.py      # writes paper/figures/C_idle_dephasing.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, NullLocator

from make_figures import COLUMN, D5, DIAG, GRID, INK, MUTED, OUT, label

V_FLOOR = 0.03  # bottom of the decay panels; points with v <= 0.05 are drawn at the 0.05 cut


def decay_panel(ax, q: dict, rounds: list[int], first: bool) -> None:
    n = np.array(rounds)
    ax.fill_between(n, q["curve_f1"], q["curve_f0"], color=GRID, alpha=0.7, lw=0)
    ax.plot(n, q["curve_f0"], color=MUTED, lw=0.6)
    ax.plot(n, q["curve_f1"], color=MUTED, lw=0.6)
    ax.plot(n, q["curve_own_f"], color=D5, lw=1.0)
    for k, (v, s, kept) in enumerate(zip(q["v"], q["sigma_v"], q["in_fit"])):
        if kept:
            ax.errorbar(n[k], v, yerr=s, fmt="o", ms=3.2, color=D5, elinewidth=0.9, capsize=0)
        else:  # not fitted: shot noise leaves nothing to fit
            ax.plot(n[k], 0.05, marker="v", ms=3.6, mfc="white", mec=D5, mew=0.8, ls="none")
    ax.set_yscale("log")
    ax.set_ylim(V_FLOOR, 1.05)
    ax.yaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1.0]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.set_yticklabels(["0.05", "0.1", "0.2", "0.5", "1"] if first else [])
    ax.set_xlim(0, 17)
    ax.set_xticks([1, 8, 16])
    ax.set_xlabel(r"idle rounds $n$")
    ax.set_title(
        f"q{q['physical_qubit']}, $T_2$ {q['t2_us']:.0f} µs",
        loc="left",
        fontsize=7,
        color=INK,
    )
    ax.text(
        0.04,
        0.05,
        f"$f = {q['f']:.2f}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        color=INK,
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.6, "alpha": 0.85},
    )
    if first:
        ax.set_ylabel(r"$v(n) = \sqrt{\langle X\rangle^2 + \langle Y\rangle^2}$")


def make(results_dir: Path, out_pdf: Path) -> None:
    data = json.loads((Path(results_dir) / "idle_f_t2.json").read_text())
    decay, t2 = data["decay"], data["f_vs_t2"]

    # Column width: the three decay panels above, f against T2 below.
    fig = plt.figure(figsize=(COLUMN, 3.4), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.15])
    top = [fig.add_subplot(grid[0, k]) for k in range(3)]
    b = fig.add_subplot(grid[1, :])

    # (a) three data qubits of one job: data, the model's band from f = 0 to f = 1, the fit
    for k, (ax, q) in enumerate(zip(top, decay["qubits"])):
        decay_panel(ax, q, decay["rounds"], first=k == 0)
    fig.legend(
        handles=[
            Line2D([], [], marker="o", ms=3.2, color=D5, ls="none", label="device"),
            Line2D([], [], color=D5, lw=1.0, label="fit"),
            Patch(color=GRID, alpha=0.7, label=r"model, $f = 0$ (top) to 1"),
        ],
        loc="outside upper center",
        ncol=3,
        handlelength=1.4,
        handletextpad=0.4,
        columnspacing=1.0,
    )
    label(top[0], "(a)")

    # (b) f against reported T2: every measured qubit-job, and each physical qubit's median
    rows, per = t2["per_qubit_job"], t2["per_physical_qubit"]
    b.axhline(1.0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    b.plot(
        [r["t2_us"] for r in rows],
        [r["f"] for r in rows],
        "o",
        ms=1.6,
        color=MUTED,
        alpha=0.45,
        mew=0,
        label=f"single job ({len(rows)})",
    )
    x = np.array([p["t2_us_median"] for p in per])
    y = np.array([p["f_median"] for p in per])
    lo, hi = np.array([p["f_quartiles"] for p in per]).T
    b.vlines(x, lo, hi, color=D5, lw=0.7, alpha=0.8)
    b.plot(
        x, y, "o", ms=3.4, color=D5, mec=INK, mew=0.4, label=f"qubit median, quartiles ({len(per)})"
    )
    b.set_xscale("log")
    b.set_xlim(4, 300)
    b.set_xticks([5, 10, 20, 50, 100, 200], ["5", "10", "20", "50", "100", "200"])
    b.xaxis.set_minor_locator(NullLocator())
    b.set_ylim(-1.0, 4.8)
    b.set_xlabel(r"reported $T_2$ (µs)")
    b.set_ylabel(r"$f$")
    rho = t2["spearman_f_t2"]
    ci = rho["ci95_bootstrap_over_qubits"]
    b.text(
        0.02,
        0.97,
        f"Spearman $\\rho = {rho['rho']:.2f}$ ({ci[0]:.2f}, {ci[1]:.2f}), {rho['points']} qubits",
        transform=b.transAxes,
        ha="left",
        va="top",
        fontsize=7,
        color=INK,
    )
    b.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), handletextpad=0.2)
    label(b, "(b)")

    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    make(DIAG, OUT / "C_idle_dephasing.pdf")
    print("wrote", OUT / "C_idle_dephasing.pdf")
