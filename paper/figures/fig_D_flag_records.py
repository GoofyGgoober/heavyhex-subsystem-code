"""The flag and relay readouts in both memory bases, and an example readout series.

delta_panel(): the lag-l persistence Delta(l) against the lag in X and Z memory, device and
model, with the fit A r^l + C (panel (a) of Fig. leak, drawn by make_figures.py). make_trace():
one flag's readouts over 120 shots of eight-round Z memory and the hidden Markov model's
posterior probability of the persistent state (App. Fig. hmm_trace). Both read only
docs/analysis/results/flags_summary.json (docs/analysis/flags_xmemory.py) and use
make_figures.py's style.

    python paper/figures/fig_D_flag_records.py   # writes hmm_trace.pdf here
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from make_figures import COLUMN, D3, D5, DIAG, INK, MUTED, OUT, label  # also sets the style

# basis: colour, marker, line style of the fit, horizontal offset (us) so the bases do not overlap
BASES = {"Z": (D3, "s", (0, (1.5, 1.5)), -0.6), "X": (D5, "o", (0, (4, 2)), 0.6)}


def load(results_dir: Path) -> dict:
    return json.loads((Path(results_dir) / "flags_summary.json").read_text())


def delta_panel(ax, results_dir: Path = DIAG) -> None:
    """Delta(l) in both bases: device with 95% intervals over qubits, model, and the fits."""
    ref = load(results_dir)["refiring"]
    for basis, (colour, marker, dash, shift) in BASES.items():
        r = ref[basis]
        t = np.array(r["lag_us"])
        y = 100 * np.array(r["device"])
        lo, hi = (100 * np.array(r["device_ci95"])).T
        f = r["fit"]
        lags = np.linspace(0.85, 7.15, 120)
        ax.plot(
            lags * t[0], 100 * (f["A"] * f["r"] ** lags + f["C"]), color=colour, lw=0.9, ls=dash
        )
        ax.errorbar(
            t + shift,
            y,
            yerr=[y - lo, hi - y],
            fmt=marker,
            ms=3.2,
            color=colour,
            elinewidth=0.8,
            capsize=0,
        )
        model = 100 * np.array(r["model_lags_1_6"])
        ax.plot(t[: len(model)] + shift, model, marker, ms=3.0, mfc="white", mec=MUTED, mew=0.7)
    ax.axhline(0, color=MUTED, lw=0.5)
    ax.set_xlabel(r"lag $\ell\tau$ (µs)")
    ax.set_ylabel(r"persistence $\Delta(\ell)$ (points)")
    ax.set_xlim(0, 60)
    ax.set_ylim(-1, 17)
    handles = [
        Line2D([], [], marker="s", ms=3.2, color=D3, ls=(0, (1.5, 1.5)), lw=0.9, label="Z memory"),
        Line2D([], [], marker="o", ms=3.2, color=D5, ls=(0, (4, 2)), lw=0.9, label="X memory"),
        Line2D([], [], marker="o", ms=3.0, mfc="white", mec=MUTED, ls="none", label="model"),
    ]
    ax.legend(handles=handles, loc="upper right", handlelength=2.4)


def make_trace(results_dir: Path = DIAG, out_pdf: Path = OUT / "hmm_trace.pdf") -> None:
    """One flag's record, shots across and rounds down, and the posterior of the persistent state."""
    ex = load(results_dir)["example_trace"]
    fired = np.array(ex["fired"]).T  # rounds x shots
    post = np.array(ex["posterior_persistent"]).T
    rounds, shots = fired.shape
    fig, (a, b) = plt.subplots(2, 1, figsize=(COLUMN, 1.75), sharex=True, layout="constrained")
    extent = (0.5, shots + 0.5, rounds + 0.5, 0.5)
    for ax, data, title in (
        (a, fired, "read 1"),
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
    b.set_xlabel("shot")
    b.set_xticks([1, 20, 40, 60, 80, 100, shots])
    fig.colorbar(im, ax=[a, b], fraction=0.04, pad=0.01, ticks=[0, 0.5, 1])
    label(a, "(a)")
    label(b, "(b)")
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    make_trace()
    print("wrote hmm_trace.pdf")
