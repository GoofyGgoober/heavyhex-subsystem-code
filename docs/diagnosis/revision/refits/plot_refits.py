"""Figure for refits.json: Λ per repetition under both estimators, and eps binomial vs WLS.

    python docs/diagnosis/revision/refits/plot_refits.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
WLS_C, MLE_C, PRED_C = "#2a78d6", "#eb6834", "#8a8a85"
INK, MUTED, GRID = "#1f1f1e", "#6b6a65", "#e6e5df"
X_C, Z_C = "#1baf7a", "#e87ba4"  # basis colours (third panel only)


def main() -> None:
    res = json.loads((HERE / "refits.json").read_text())
    reps = res["per_rep"]
    keys = sorted(reps, key=lambda k: (reps[k]["placement"], reps[k]["day"]))
    plt.rcParams.update({"font.size": 8.5, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.4), gridspec_kw={"width_ratios": [1.2, 1.2, 1]})
    x = list(range(len(keys)))
    labels = [f"P{reps[k]['placement']} d{int(reps[k]['day'][-2:]) - 3}" for k in keys]
    for ax, b in zip(axes[:2], "XZ"):
        lw = [reps[k][b]["lambda"]["wls"] for k in keys]
        lm = [reps[k][b]["lambda"] for k in keys]
        pr = [reps[k][b]["predicted"]["lambda"]["binomial"] for k in keys]
        ax.errorbar([i - 0.15 for i in x], [v for v, _ in lw], yerr=[s for _, s in lw], fmt="o", ms=5,
                    color=WLS_C, lw=1.5, capsize=0, label="weighted-log fit (registered) ±1σ")
        ax.errorbar([i + 0.15 for i in x], [m["binomial"] for m in lm],
                    yerr=[[m["binomial"] - m["binomial_lo68_quasi"] for m in lm],
                          [m["binomial_hi68_quasi"] - m["binomial"] for m in lm]],
                    fmt="s", ms=5, color=MLE_C, lw=1.5, capsize=0, label="binomial MLE, 68% profile (quasi)")
        ax.plot(x, pr, "D", ms=5, mfc="none", mec=PRED_C, mew=1.2, label="prediction, binomial MLE")
        ax.axvline(4.5, color=GRID, lw=1)
        ax.set_xticks(x, labels, rotation=60)
        ax.set_ylabel(f"Λ_{b} = ε₃/ε₅")
        ax.set_title(f"{b} memory", color=INK, loc="left")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.set_axisbelow(True)
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, frameon=False, fontsize=8, loc="upper left", ncol=3, bbox_to_anchor=(0.06, 1.0))
    ax = axes[2]
    curves = res["curves"]
    for b, mk, col in (("X", "o", X_C), ("Z", "^", Z_C)):
        for d in ("3", "5"):
            sel = [c for k, c in curves.items() if f" {b} d{d}" in k]
            ax.plot([100 * c["wls"]["eps"] for c in sel], [100 * c["binomial"]["eps"] for c in sel], mk,
                    ms=5, mfc="none" if d == "3" else col, mec=col, mew=1.2, label=f"{b} memory, d={d}")
    lim = [8, 37]
    ax.plot(lim, lim, color=MUTED, lw=0.8, zorder=0)
    ax.set_xlim(lim), ax.set_ylim(lim)
    ax.set_xlabel("ε, weighted-log fit (%)")
    ax.set_ylabel("ε, binomial MLE (%)")
    ax.set_title("each of the 40 curves", color=INK, loc="left")
    ax.grid(color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=7.5)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(HERE / "refits.png", dpi=180)
    fig.savefig(HERE / "refits.pdf")


if __name__ == "__main__":
    main()
