"""Detection-event figures: the fraction per round (main text) and p_ij (appendix).

    python paper/figures/fig_B_detector_diagnostics.py

make(results_dir, out_pdf) draws the detection-event fraction against the round from
results/detection_events.json (B_detector_diagnostics.pdf); make_pij(results_dir, out_pdf) the
p_ij matrix, the learned edge probabilities and the pairs the model never connects from
results/pij.json (B_detector_diagnostics_pij.pdf). The style is make_figures.py's (imported).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_figures import COLUMN, D3, D5, DIAG, GRID, INK, MUTED, OUT, PAGE, label

# Detector classes: marker for the device, dash for the model.
STYLE = {
    "X weight 4": ("o", "-", "weight-4 stabilizers"),
    "X weight 2": ("s", (0, (4, 2)), "weight-2 stabilizers"),
    "Z": ("o", "-", "stabilizers"),
    "flag": ("s", (0, (4, 2)), "flags"),
    "relay": ("^", (0, (1.2, 1.4)), "relays"),
}
CLASSES = {"X": ("X weight 4", "X weight 2"), "Z": ("Z", "flag", "relay")}


def _load(results_dir: Path, name: str) -> dict:
    return json.loads((Path(results_dir) / name).read_text())


def _fill(marker: str) -> str:
    return D5 if marker == "o" else "white"


def make(results_dir: Path = DIAG, out_pdf: Path = OUT / "B_detector_diagnostics.pdf") -> None:
    """Detection-event fraction per round, 8-round circuits, mean over the ten repetitions."""
    curves = _load(results_dir, "detection_events.json")["rounds"]
    fig, axes = plt.subplots(2, 2, figsize=(COLUMN, 3.0), sharex=True)
    for row, basis in enumerate("XZ"):
        for col, d in enumerate("35"):
            ax = axes[row, col]
            for c in CLASSES[basis]:
                marker, dash, _ = STYLE[c]
                k = curves[f"{basis} d{d} {c} placement both"]
                t = np.array(k["t"])
                dev, mod = np.array(k["device"]), np.array(k["model"])
                ax.plot(t, mod[:, 0], color=MUTED, lw=0.9, ls=dash)
                ax.errorbar(
                    t,
                    dev[:, 0],
                    yerr=dev[:, 1],
                    fmt=marker,
                    ms=2.8,
                    color=D5,
                    mfc=_fill(marker),
                    mew=0.8,
                    elinewidth=0.7,
                    capsize=0,
                )
            ax.axvline(8.5, color=GRID, lw=0.6)  # t = 9 is the final layer, from the data readout
            ax.set_xticks(range(1, 10))
            ax.set_xlim(0.5, 9.5)
            ax.set_ylim(0, None)
            ax.grid(axis="y", color=GRID, lw=0.4)
            ax.set_title(f"{basis} memory, $d={d}$", loc="left", fontsize=8, color=INK)
            if col == 0:
                ax.set_ylabel("detection-event fraction")
            if row == 1:
                ax.set_xlabel("round $t$")
    # Each legend in an empty band: X classes below the data of (b), Z classes mid-height in (d).
    for (row, col, where), basis in zip(((0, 1, "lower right"), (1, 1, "center right")), "XZ"):
        handles = [
            Line2D(
                [],
                [],
                marker=STYLE[c][0],
                ms=2.8,
                mec=D5,
                mfc=_fill(STYLE[c][0]),
                color=MUTED,
                lw=0.9,
                ls=STYLE[c][1],
                label=STYLE[c][2],
            )
            for c in CLASSES[basis]
        ]
        axes[row, col].legend(
            handles=handles, loc=where, handlelength=2.2, handletextpad=0.3, borderaxespad=0.2
        )
    axes[0, 0].legend(
        handles=[
            Line2D([], [], marker="o", ms=2.8, color=D5, ls="none", label="device"),
            Line2D([], [], color=MUTED, lw=0.9, label="model"),
        ],
        loc="lower right",
        handletextpad=0.3,
        borderaxespad=0.2,
    )
    for ax, tag in zip(axes.flat, "abcd"):
        label(ax, f"({tag})")
    fig.tight_layout(w_pad=0.6, h_pad=0.6)
    fig.savefig(out_pdf)
    plt.close(fig)


CMAP = LinearSegmentedColormap.from_list("p", ["#ffffff", D5, INK])
VMAX = 0.12
EDGE = {  # class: marker, colour
    "space-like": ("o", D5),
    "time-like": ("s", D3),
    "diagonal": ("D", INK),
    "boundary": ("v", MUTED),
    "flag/relay": ("^", "#9a9995"),
}


def _matrix(ax, m: dict):
    """Device p_ij above the diagonal, model below; detectors ordered by round."""
    labels = m["labels"]
    idx = sorted(range(len(labels)), key=lambda a: (labels[a][1], labels[a][2]))
    dev = np.array(m["device"], float)[np.ix_(idx, idx)]
    mod = np.array(m["model"], float)[np.ix_(idx, idx)]
    n = len(idx)
    shown = np.where(np.triu(np.ones((n, n), bool), 1), dev, mod)
    np.fill_diagonal(shown, np.nan)
    im = ax.imshow(np.clip(shown, 0, None), cmap=CMAP, vmin=0, vmax=VMAX, interpolation="nearest")
    rounds = [labels[a][1] for a in idx]
    starts = [0] + [j for j in range(1, n) if rounds[j] != rounds[j - 1]]
    for j in starts[1:]:
        ax.axhline(j - 0.5, color=GRID, lw=0.3)
        ax.axvline(j - 0.5, color=GRID, lw=0.3)
    mid = [(s + e) / 2 - 0.5 for s, e in zip(starts, [*starts[1:], n])]
    ax.set_xticks(mid, [str(r + 1) for r in sorted(set(rounds))])
    ax.set_yticks(mid, [str(r + 1) for r in sorted(set(rounds))])
    ax.tick_params(length=0, labelsize=6)
    ax.set_xlabel("round $t$")
    ax.text(0.97, 0.97, "device", transform=ax.transAxes, ha="right", va="top", fontsize=7)
    ax.text(0.03, 0.03, "model", transform=ax.transAxes, ha="left", va="bottom", fontsize=7)
    return im


def make_pij(
    results_dir: Path = DIAG, out_pdf: Path = OUT / "B_detector_diagnostics_pij.pdf"
) -> None:
    """(a) X-memory p_ij, placement 1's distance-3 patch, eight rounds, mean over the five days;
    (b) device against model p per edge type, all jobs; (c) pairs the model never connects."""
    r = _load(results_dir, "pij.json")
    fig, (a, b, c) = plt.subplots(
        1, 3, figsize=(PAGE, 2.3), width_ratios=[1, 1.05, 0.9], layout="constrained"
    )
    m = next(v for k, v in r.items() if k.startswith("matrix"))["X"]
    im = _matrix(a, m)
    cb = fig.colorbar(im, ax=a, fraction=0.05, pad=0.02, extend="max")
    cb.set_label("$p_{ij}$")
    cb.outline.set_linewidth(0.4)
    cb.ax.tick_params(labelsize=6)
    a.set_title("X memory, $d=3$", loc="left", fontsize=8, color=INK)

    for cls, (marker, colour) in EDGE.items():
        x, y = [], []
        for j in r["per_job"].values():
            for t in j["types"].values():
                if t[0] == cls and t[5] is not None and t[5] > 0 and t[4] > 0:
                    x.append(t[4])
                    y.append(t[5])
        b.plot(x, y, marker, ms=1.6, color=colour, alpha=0.35, mew=0, ls="none")
    lim = (1e-3, 0.5)
    b.plot(lim, lim, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_xlim(*lim)
    b.set_ylim(*lim)
    b.set_xlabel("model $p$")
    b.set_ylabel("device $p$ (learned)")
    b.legend(
        handles=[
            Line2D([], [], marker=mk, ms=3.2, color=col, ls="none", label=cls)
            for cls, (mk, col) in EDGE.items()
        ],
        loc="lower right",
        handletextpad=0.1,
        borderaxespad=0.1,
        labelspacing=0.25,
    )

    pairs = next(v for k, v in r.items() if k.startswith("unmodelled"))
    lags = np.array([0, 1, 2, 3])
    for basis, colour in (("X", D5), ("Z", D3)):
        for same, marker, name in (
            ("same", "o", "same detector"),
            ("other", "s", "other detectors"),
        ):
            rows = [pairs.get(f"both {basis} {same} {lag}") for lag in lags]
            keep = [k for k, row in enumerate(rows) if row]
            if not keep:
                continue
            dev = np.array([rows[k]["device_mean_p"] for k in keep])
            ctl = np.array([rows[k]["control_mean_p"] for k in keep])
            shift = (-0.12 if basis == "X" else 0.12) + (-0.04 if same == "same" else 0.04)
            c.errorbar(
                lags[keep] + shift,
                dev[:, 0],
                yerr=dev[:, 1],
                fmt=marker + ("-" if same == "same" else "--"),
                ms=3.0,
                lw=0.8,
                color=colour,
                mfc=colour if same == "same" else "white",
                elinewidth=0.7,
                capsize=0,
                label=f"{basis}, {name}",
            )
            c.plot(lags[keep] + shift, ctl[:, 0], marker, ms=2.4, color=MUTED, mfc="none", mew=0.5)
    c.axhline(0, color=MUTED, lw=0.5)
    c.set_xticks(lags, ["0", "1", "2", "3+"])
    c.set_xlabel("rounds apart")
    c.set_ylabel("mean $p_{ij}$, pairs with no model edge")
    c.set_ylim(-0.002, 0.06)  # room above the data for the legend
    c.legend(
        loc="upper center",
        ncol=2,
        handlelength=1.8,
        handletextpad=0.3,
        labelspacing=0.25,
        columnspacing=0.8,
        fontsize=6,
    )
    for ax, tag in zip((a, b, c), "abc"):
        label(ax, f"({tag})")
    fig.savefig(out_pdf)
    plt.close(fig)


if __name__ == "__main__":
    make()
    make_pij()
    print("wrote B_detector_diagnostics.pdf, B_detector_diagnostics_pij.pdf")
