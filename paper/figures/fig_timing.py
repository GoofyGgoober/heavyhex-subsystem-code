"""Fig. timing: one round of X memory on Fez, qubit by qubit, and what sets its length.

Top: the middle round of day 1's 3-round X-memory circuit on pair 1's distance-3 patch, as the
compiled circuit ran it, for one qubit of each role (the gadget figure's q24, q23 and q16, and
the relay q21). Bottom: the round's critical path, the same with resets that take no time
(no_reset.py's stand-in for a reset-free schedule), and the parallel, reset-free round of Harper
et al. on Heron r2. Draws from docs/analysis/results/timing.json (docs/analysis/timing.py) in
the style of make_figures.py, whose fonts and rcParams it imports. Run from the repository root:

    python paper/figures/fig_timing.py      # writes paper/figures/timing.pdf
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from make_figures import COLUMN, D5, DIAG, GRID, INK, MUTED, OUT

HARPER_ROUND_US = 3.2  # Harper et al., arXiv:2504.07258: parallel, reset-free round on Heron r2
ROWS = {  # role in timing.json: row label
    "data": "data q24",
    "flag": "X anc., flag q23",
    "z_ancilla": "Z ancilla q16",
    "relay": "relay q21",
}
HEIGHT = 0.62  # of a bar, in row units
STYLE = {  # kind: face colour, hatch, height as a share of the bar
    "readout": (D5, None, 1.0),
    "reset": ("white", "//////", 1.0),
    "cz": (INK, None, 1.0),
    "single_qubit": (INK, None, 0.5),  # short: like the X pulses, single-qubit gates
    "idle_decoupled": (GRID, None, 1.0),
    "short_wait": (None, None, 0.0),
    "gates": (INK, None, 1.0),  # on the critical path: the gates and the short waits among them
}


def box(ax, kind: str, x: float, width: float, y: float) -> None:
    face, hatch, share = STYLE[kind]
    if face is None:
        return
    h = HEIGHT * share
    ax.add_patch(
        Rectangle(
            (x, y - h / 2),
            width,
            h,
            facecolor=face,
            edgecolor=D5 if hatch else "none",
            hatch=hatch,
            lw=0.4 if hatch else 0,
        )
    )


def make(results_dir: Path, out_pdf: Path) -> None:
    data = json.loads((Path(results_dir) / "timing.json").read_text())
    patch = data["patches"]["3"]
    round_us = data["round_us"]
    mid_us = patch["halves"][0]["us"]

    fig = plt.figure(figsize=(COLUMN, 2.15), layout="constrained")
    top, bottom = fig.subplots(2, 1, sharex=True, height_ratios=[4, 3])
    with mpl.rc_context({"hatch.linewidth": 0.5}):
        # Top: one qubit per role, first row at the top.
        for row, role in enumerate(ROWS):
            q = patch["drawn"][role]
            y = -row
            for kind, start, width in q["segments_ns"]:
                box(top, kind, start / 1e3, width / 1e3, y)
            for t in q["x_pulses_ns"]:  # the decoupling pulses inside the idle periods
                box(top, "single_qubit", t / 1e3, data["durations_ns"]["x"] / 1e3, y)
        top.axvline(mid_us, color=MUTED, lw=0.6, ls=(0, (3, 2)))
        top.set_yticks([-row for row in range(len(ROWS))], list(ROWS.values()))
        top.set_ylim(-len(ROWS) + 0.45, 0.85)
        for x, text in ((mid_us / 2, "X gauges"), ((mid_us + round_us) / 2, "Z gauges")):
            top.text(x, 0.48, text, ha="center", va="bottom", fontsize=7, color=INK)

        # Bottom: what sets the round's length.
        bars = [
            ("as run", [(k, w) for k, w in data["critical_path_us"]]),
            ("zero-time resets", [(k, w) for k, w in data["critical_path_us"] if k != "reset"]),
        ]
        for row, (_, path) in enumerate(bars):
            x = 0.0
            for kind, width in path:
                box(bottom, kind, x, width, -row)
                x += width
            bottom.text(
                x - 0.08, -row, f"{x:.1f} µs", ha="right", va="center", fontsize=7, color="white"
            )
        y = -len(bars)
        bottom.add_patch(
            Rectangle(
                (0, y - HEIGHT / 2),
                HARPER_ROUND_US,
                HEIGHT,
                facecolor="white",
                edgecolor=MUTED,
                lw=0.6,
            )
        )
        bottom.text(
            HARPER_ROUND_US - 0.08,
            y,
            f"{HARPER_ROUND_US:.1f} µs",
            ha="right",
            va="center",
            fontsize=7,
            color=INK,
        )
        bottom.set_yticks(
            [-row for row in range(len(bars) + 1)],
            [name for name, _ in bars] + ["Harper et al."],
        )
        bottom.set_ylim(-len(bars) - 0.55, 0.55)
        bottom.set_xlim(0, round_us)
        bottom.set_xticks([0, 2, 4, 6, round_us], ["0", "2", "4", "6", f"{round_us:.1f}"])
        bottom.set_xlabel("time in the round (µs)")

        for ax in (top, bottom):
            ax.tick_params(axis="y", length=0)
            ax.spines["left"].set_visible(False)
        top.tick_params(axis="x", length=0)
        top.spines["bottom"].set_visible(False)

        handles = [
            Patch(facecolor=D5, label="readout"),
            Patch(facecolor="white", edgecolor=D5, hatch="//////", lw=0.4, label="reset"),
            Patch(facecolor=INK, label="CZ"),
            Line2D([], [], color=INK, lw=0, marker="|", ms=3.5, mew=0.9, label=r"$\sqrt{X}$, $X$"),
            Patch(facecolor=GRID, label="decoupled idle"),
        ]
        fig.legend(
            handles=handles,
            loc="outside lower center",
            ncol=5,
            handlelength=1.0,
            handleheight=0.8,
            handletextpad=0.3,
            columnspacing=0.8,
        )
        fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    make(DIAG, OUT / "timing.pdf")
    print("wrote", OUT / "timing.pdf")
