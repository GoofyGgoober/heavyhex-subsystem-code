"""Fig. A_effect_sizes: observed over predicted for each quantity, per pair and over the ten
repetitions (binomial fit), from docs/analysis/results/effect_sizes.json.

    python paper/figures/fig_A_effect_sizes.py      # writes paper/figures/A_effect_sizes.pdf

Style, fonts and colours are make_figures.py's (imported when drawing); it needs Latin Modern
as make_figures.py does.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROWS = (
    ("X", "X eps3", r"$\varepsilon_3$"),
    ("X", "X eps5", r"$\varepsilon_5$"),
    ("X", "X lambda", r"$\Lambda_\mathrm{X}$"),
    ("Z", "Z eps3", r"$\varepsilon_3$"),
    ("Z", "Z eps5", r"$\varepsilon_5$"),
)  # no Lambda_Z: distance-5 Z memory is close to random (Supplementary, Robustness)


def make(results_dir: Path, out_pdf: Path) -> None:
    sys.path.insert(0, str(HERE))
    import make_figures as base  # rcParams, font and palette of the paper's figures

    plt, Line2D = base.plt, base.Line2D
    data = json.loads((Path(results_dir) / "effect_sizes.json").read_text())["ratios"]["binomial"]
    # Per pair: geometric mean over its five days, t (4 dof) 95% interval. All ten:
    # random-effects mean of ln(O/M), Hartung-Knapp 95% interval with its variance factor
    # floored at one. Open marks for the pairs, a filled diamond for the ten; shape alone
    # separates them in greyscale.
    open_mark = {"mfc": "white", "mec": base.MUTED, "mew": 0.8}
    marks = (
        ("pair 1", +0.24, {"marker": "o", "ms": 3.4, **open_mark}),
        ("pair 2", 0.0, {"marker": "s", "ms": 3.2, **open_mark}),
        ("all ten", -0.24, {"marker": "D", "ms": 3.6, "mfc": base.D5, "mec": base.D5, "mew": 0.8}),
    )
    fig, ax = plt.subplots(figsize=(base.COLUMN, 2.1), layout="constrained")
    y0 = {"X": 3.6, "Z": 0.0}  # a gap between the two blocks for their headings
    count = {b: sum(r[0] == b for r in ROWS) for b in y0}
    ys, labels = [], []
    for basis, key, label in ROWS:
        y = y0[basis] + count[basis] - 1 - [r[1] for r in ROWS if r[0] == basis].index(key)
        ys.append(y)
        labels.append(label)
        for name, dy, style in marks:
            v = data[key][name.replace("pair", "placement")]  # pair i's d = 5 patch is placement i
            ci = v["ci95_factor_at_least_1"] if name == "all ten" else v["ci95"]
            colour = base.D5 if name == "all ten" else base.MUTED
            ax.plot(ci, [y + dy] * 2, color=colour, lw=0.9, solid_capstyle="butt")
            ax.plot(v["ratio"], y + dy, ls="none", **style)
    for basis, top in y0.items():
        heading = top + count[basis] - 0.25
        ax.text(
            0.77, heading, f"{basis} memory", ha="left", va="center", fontsize=7, style="italic"
        )
    ax.axvline(1.0, color=base.MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.set_xscale("log")
    ticks = [0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.75, 3.3)
    ax.set_yticks(ys, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, y0["X"] + count["X"] + 0.2)
    ax.grid(axis="x", color=base.GRID, lw=0.4)
    ax.set_axisbelow(True)
    ax.set_xlabel("observed / predicted (95% interval)")
    handles = [Line2D([], [], ls="none", label=name, **style) for name, _, style in marks]
    ax.legend(handles=handles, loc="upper right", handletextpad=0.2, borderaxespad=0.1)
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    make(HERE.parents[1] / "docs" / "analysis" / "results", HERE / "A_effect_sizes.pdf")
    print("wrote", HERE / "A_effect_sizes.pdf")
