"""Fig. flag_records: one flag's readouts and the persistence of the flag and relay readouts.

(a) One flag's readouts over 120 shots of eight-round Z memory and the hidden Markov model's
posterior probability of the persistent state. (b) The lag-l persistence Delta(l) against the
lag in rounds, in X and Z memory, device and model, with the fit Delta_0 kappa^l + Delta_inf.
Reads only docs/analysis/results/flags_summary.json (docs/analysis/flags_xmemory.py) and uses
make_figures.py's style.

    python paper/figures/fig_D_flag_records.py   # writes flag_records.pdf here
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D

from make_figures import D3, D5, DIAG, INK, MUTED, OUT, PAGE, label  # also sets the style

# basis: colour, marker, line style of the fit, horizontal offset (rounds) so the bases do not
# overlap
BASES = {"Z": (D3, "s", (0, (1.5, 1.5)), -0.08), "X": (D5, "o", (0, (4, 2)), 0.08)}


def load(results_dir: Path) -> dict:
    return json.loads((Path(results_dir) / "flags_summary.json").read_text())


def trace_panels(a, b, fig, results_dir: Path) -> None:
    """One flag's record, shots across and rounds down, and the posterior of the persistent state."""
    ex = load(results_dir)["example_trace"]
    fired = np.array(ex["fired"]).T  # rounds x shots
    post = np.array(ex["posterior_persistent"]).T
    rounds, shots = fired.shape
    extent = (0.5, shots + 0.5, rounds + 0.5, 0.5)
    for ax, data, title in (
        (a, fired, "reads 1 (black)"),
        (b, post, "posterior probability of the persistent state"),
    ):
        im = ax.imshow(
            data,
            cmap="Greys",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="nearest",
            extent=extent,
        )
        ax.set_yticks([1, rounds])
        ax.set_ylabel("round")
        ax.set_title(title, loc="left", fontsize=7, color=INK, pad=2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(True)
    a.tick_params(axis="x", labelbottom=False)
    b.set_xlabel("shot")
    b.set_xticks([1, 20, 40, 60, 80, 100, shots])
    fig.colorbar(im, ax=[a, b], fraction=0.04, pad=0.01, ticks=[0, 0.5, 1])


def delta_panel(ax, results_dir: Path) -> None:
    """Delta(l) in both bases: device with 95% intervals over qubits, model, and the fits."""
    ref = load(results_dir)["refiring"]
    for basis, (colour, marker, dash, shift) in BASES.items():
        r = ref[basis]
        lags = np.array(r["lags"])
        y = 100 * np.array(r["device"])
        lo, hi = (100 * np.array(r["device_ci95"])).T
        f = r["fit"]
        smooth = np.linspace(0.85, 7.15, 120)
        ax.plot(smooth, 100 * (f["A"] * f["r"] ** smooth + f["C"]), color=colour, lw=0.9, ls=dash)
        ax.errorbar(
            lags + shift,
            y,
            yerr=[y - lo, hi - y],
            fmt=marker,
            ms=3.2,
            color=colour,
            elinewidth=0.8,
            capsize=0,
        )
        model = 100 * np.array(r["model_lags_1_6"])
        ax.plot(lags[: len(model)] + shift, model, marker, ms=3.0, mfc="white", mec=MUTED, mew=0.7)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set_xlabel(r"lag $\ell$ (rounds)")
    ax.set_ylabel(r"$\Delta(\ell)$ (percentage points)")
    ax.set_xticks(range(1, 8))
    ax.set_xlim(0.5, 7.5)
    ax.set_ylim(-1, 17)
    handles = [
        Line2D([], [], marker="s", ms=3.2, color=D3, ls=(0, (1.5, 1.5)), lw=0.9),
        Line2D([], [], marker="o", ms=3.2, color=D5, ls=(0, (4, 2)), lw=0.9),
        tuple(Line2D([], [], marker=m, ms=3.0, mfc="white", mec=MUTED, ls="none") for m in "so"),
    ]
    ax.legend(
        handles,
        ["Z memory", "X memory", "model"],
        loc="upper right",
        handlelength=2.4,
        handler_map={tuple: HandlerTuple(ndivide=None)},
    )


def make(results_dir: Path = DIAG, out_pdf: Path = OUT / "flag_records.pdf") -> None:
    fig = plt.figure(figsize=(PAGE, 1.85), layout="constrained")
    grid = fig.add_gridspec(2, 2, width_ratios=[1.55, 1])
    a = fig.add_subplot(grid[0, 0])
    b = fig.add_subplot(grid[1, 0], sharex=a)
    c = fig.add_subplot(grid[:, 1])
    trace_panels(a, b, fig, results_dir)
    delta_panel(c, results_dir)
    label(a, "(a)")
    label(c, "(b)")
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    make()
    print("wrote flag_records.pdf")
