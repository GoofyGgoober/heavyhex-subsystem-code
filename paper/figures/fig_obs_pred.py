"""Fig. observed against predicted: the logical error per round of all 40 curves.

One point per curve (two bases, two distances, ten pair-days): the device's eps from the
binomial fit, with its 68% profile interval, against the prediction stored before the job at
f = 1 (filled), and against the prediction at the job's measured f (open, joined to it). The
predictions' own shot noise is smaller than the markers. Draws from docs/analysis/results/
fits.json and effect_sizes.json ("prejob") in the style of make_figures.py, whose fonts and
rcParams it imports. Run from the repository root:

    python paper/figures/fig_obs_pred.py      # writes paper/figures/obs_pred.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullFormatter

from make_figures import COLUMN, D3, D5, DAYS, DIAG, GRID, INK, MUTED, OUT, label

LIMITS = (5.0, 42.0)  # both axes, eps in percent
TICKS = [5, 10, 20, 40]
STYLE = {"3": (D3, "s", 3.0), "5": (D5, "o", 3.4)}  # colour, marker, size; as in Fig. pn


def curves(results_dir: Path) -> list[dict]:
    """The 40 curves: observed eps and its 68% interval, the frozen f = 1 and measured-f
    predictions, all from the binomial fit."""
    fits = json.loads((results_dir / "fits.json").read_text())
    rows = json.loads((results_dir / "effect_sizes.json").read_text())["prejob"]["binomial"]
    prejob = {(r["day"], r["placement"]): r for r in rows["per_repetition"]}
    out = []
    for rep in fits["per_rep"].values():
        frozen = prejob[(DAYS.index(rep["day"]) + 1, rep["placement"])]
        for basis in "XZ":
            for d in "35":
                obs = rep[basis][f"eps{d}"]
                pred = fits["predictions"][rep["jobs"][d]]
                at1 = pred["frozen_f1.0"][basis][d]["binomial"]["eps"]
                assert at1 == frozen[f"{basis} eps{d} frozen_f1.0"]
                assert obs["binomial"] == frozen[f"{basis} eps{d} observed"]
                out.append(
                    {
                        "basis": basis,
                        "d": d,
                        "obs": obs["binomial"],
                        "lo": obs["binomial_lo68_quasi"],
                        "hi": obs["binomial_hi68_quasi"],
                        "at_f1": at1,
                        "at_fhat": pred["predicted_at_measured_dephasing"][basis][d]["binomial"][
                            "eps"
                        ],
                    }
                )
    assert len(out) == 40
    return out


def panel(ax, rows: list[dict], basis: str) -> None:
    ax.plot(LIMITS, LIMITS, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
    for d, (colour, marker, ms) in STYLE.items():
        sel = [r for r in rows if r["basis"] == basis and r["d"] == d]
        y = 100 * np.array([r["obs"] for r in sel])
        lo = 100 * np.array([r["lo"] for r in sel])
        hi = 100 * np.array([r["hi"] for r in sel])
        x1 = 100 * np.array([r["at_f1"] for r in sel])
        xf = 100 * np.array([r["at_fhat"] for r in sel])
        # the measured-f prediction: open, light, joined to the frozen one at the same height
        ax.hlines(y, np.minimum(x1, xf), np.maximum(x1, xf), color=colour, lw=0.5, alpha=0.6)
        ax.plot(xf, y, marker, ms=ms * 0.85, mfc="white", mec=colour, mew=0.6, ls="none", alpha=0.8)
        ax.errorbar(
            x1,
            y,
            yerr=[y - lo, hi - y],
            fmt=marker,
            ms=ms,
            color=colour,
            mec="white",
            mew=0.3,
            elinewidth=0.7,
            capsize=0,
            zorder=3,
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(LIMITS)
    ax.set_ylim(LIMITS)
    ax.set_aspect("equal")
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(FixedLocator(TICKS))
        axis.set_minor_locator(FixedLocator([6, 7, 8, 9, 30]))
        axis.set_minor_formatter(NullFormatter())
    ax.set_xticklabels([str(t) for t in TICKS])
    ax.tick_params(which="minor", length=1.5, width=0.5)
    ax.grid(which="major", color=GRID, lw=0.4)
    ax.set_title(f"{basis} memory", loc="left", fontsize=8, color=INK)
    ax.set_xlabel(r"predicted $\varepsilon$ (%)")


def make(results_dir: Path, out_pdf: Path) -> None:
    rows = curves(Path(results_dir))
    fig, axes = plt.subplots(1, 2, figsize=(COLUMN, 2.05), sharey=True, layout="constrained")
    for ax, basis in zip(axes, "XZ"):
        panel(ax, rows, basis)
    axes[0].set_yticklabels([str(t) for t in TICKS])
    axes[0].set_ylabel(r"observed $\varepsilon$ (%)")
    axes[0].text(
        33, 26, "$y = x$", rotation=45, ha="center", va="center", fontsize=6.5, color=MUTED
    )
    for ax, tag in zip(axes, "ab"):
        label(ax, f"({tag})")
    fig.legend(
        handles=[
            Line2D([], [], marker="s", ms=3.0, color=D3, ls="none", label="$d=3$"),
            Line2D([], [], marker="o", ms=3.4, color=D5, ls="none", label="$d=5$"),
            Line2D([], [], marker="o", ms=3.4, color=MUTED, ls="none", label="at $f=1$"),
            Line2D(
                [],
                [],
                marker="o",
                ms=2.9,
                mfc="white",
                mec=MUTED,
                mew=0.6,
                ls="none",
                label=r"at $\hat f$",
            ),
        ],
        loc="outside upper center",
        ncol=4,
        handletextpad=0.2,
        columnspacing=1.0,
    )
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    make(DIAG, OUT / "obs_pred.pdf")
    print("wrote", OUT / "obs_pred.pdf")
