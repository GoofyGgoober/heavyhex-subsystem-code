"""Detection-event figure: the fraction per round, device against model, by detector class.

    python paper/figures/fig_B_detector_diagnostics.py

make(results_dir, out_pdf) draws the detection-event fraction against the round from
results/detection_events.json (B_detector_diagnostics.pdf). The style is make_figures.py's
(imported), and each class takes the colour of the qubits that measure it in Fig. 1.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, NullLocator

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import (
    COLUMN,
    DIAG,
    GRID,
    INK,
    MUTED,
    OUT,
    RELAY_ROLE,
    X_ROLE,
    Z_ROLE,
    label,
)

# Detector classes: colour, marker, filled, model line, legend label. The device is the markers,
# the model the line of the same colour.
SOLID, DASHED = "-", (0, (3, 1.5))
STYLE = {
    "X weight 4": (X_ROLE, "o", True, SOLID, "weight-4 X stabilizers"),
    "X weight 2": (X_ROLE, "s", False, DASHED, "weight-2 X stabilizers"),
    "Z": (Z_ROLE, "o", True, SOLID, "Z stabilizers"),
    "flag": (X_ROLE, "^", False, DASHED, "flags"),
    "relay": (RELAY_ROLE, "D", False, SOLID, "relays"),
}
CLASSES = {"X": ("X weight 4", "X weight 2"), "Z": ("Z", "flag", "relay")}
TICKS = [0.03, 0.05, 0.1, 0.2, 0.5]


def _load(results_dir: Path, name: str) -> dict:
    return json.loads((Path(results_dir) / name).read_text())


def make(results_dir: Path = DIAG, out_pdf: Path = OUT / "B_detector_diagnostics.pdf") -> None:
    """Detection-event fraction per round, 8-round circuits, mean over the ten repetitions."""
    curves = _load(results_dir, "detection_events.json")["rounds"]
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN, 3.2), sharex=True, layout="constrained")
    for row, basis in enumerate("XZ"):
        for col, d in enumerate("35"):
            ax = axes[row, col]
            for c in CLASSES[basis]:
                colour, marker, filled, line, _ = STYLE[c]
                k = curves[f"{basis} d{d} {c} placement both"]
                # Flags and relays lie close together: set them a little apart in t.
                t = np.array(k["t"]) + {"flag": -0.15, "relay": 0.15}.get(c, 0.0)
                dev, mod = np.array(k["device"]), np.array(k["model"])
                ax.plot(t, mod[:, 0], color=colour, lw=0.9, ls=line, alpha=0.85)
                ax.errorbar(
                    t,
                    dev[:, 0],
                    yerr=dev[:, 1],
                    fmt=marker,
                    ms=2.8,
                    color=colour,
                    mfc=colour if filled else "white",
                    mec=colour,
                    mew=0.7,
                    elinewidth=0.7,
                    capsize=0,
                )
            ax.axvline(8.5, color=GRID, lw=0.6)  # t = 9 is the final layer, from the data readout
            ax.set_xticks(range(1, 10))
            ax.set_xlim(0.5, 9.5)
            ax.set_yscale("log")
            ax.set_ylim(0.03, 0.6)
            ax.yaxis.set_major_locator(FixedLocator(TICKS))
            ax.yaxis.set_minor_locator(NullLocator())
            ax.set_yticklabels([f"{v:g}" for v in TICKS] if col == 0 else [])
            ax.grid(axis="y", color=GRID, lw=0.4)
            ax.set_title(f"{basis} memory, $d={d}$", loc="left", fontsize=8, color=INK)
            if col == 0:
                ax.set_ylabel("detection-event fraction")
            if row == 1:
                ax.set_xlabel("round $t$")
    handles = [
        Line2D(
            [],
            [],
            marker=m,
            ms=3.0,
            color=c,
            mfc=c if filled else "white",
            mew=0.7,
            ls="none",
            label=name,
        )
        for c, m, filled, _, name in STYLE.values()
    ] + [
        Line2D([], [], marker="o", ms=3.0, color=MUTED, ls="none", label="device"),
        Line2D([], [], color=MUTED, lw=0.9, label="model"),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=3,
        handlelength=1.4,
        handletextpad=0.3,
        columnspacing=0.8,
        fontsize=6.5,
    )
    for ax, tag in zip(axes.flat, "abcd"):
        label(ax, f"({tag})")
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    make()
    print("wrote B_detector_diagnostics.pdf")
