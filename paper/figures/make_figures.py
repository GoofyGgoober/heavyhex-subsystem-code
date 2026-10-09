"""Draw the paper's figures from the committed run folders and diagnosis results.

Run from the repository root:  python paper/figures/make_figures.py
Writes vector PDFs next to this file. Reads only runs/ and docs/diagnosis/results/.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
DIAG = ROOT / "docs" / "diagnosis" / "results"
OUT = Path(__file__).resolve().parent

DAYS = ["2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"]
ROUNDS = [1, 2, 3, 4, 6, 8]
ROUND_US = 7.9  # one round at both distances

# Two categorical slots, validated for colour-blind separation; d = 5 is the subject.
D3, D5 = "#eb6834", "#2a78d6"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COLUMN, PAGE = 3.375, 7.0  # revtex column and page widths, inches

mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Latin Modern Roman", "CMU Serif", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 8,
        "legend.fontsize": 7,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.edgecolor": MUTED,
        "axes.labelcolor": INK,
        "axes.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.2,
        "pdf.fonttype": 42,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    }
)


def load(path: Path):
    return json.loads(path.read_text())


def folders(rep: int, distance: int, day: str) -> tuple[str, int]:
    """The run folder holding this patch on this day, and its index in that folder."""
    if rep == 1:
        return f"ibm_fez-{day}-r1", (0 if distance == 3 else 1)
    return f"ibm_fez-{day}-r2-d{distance}", 0


def p_of_n(rep: int, distance: int, basis: str, which: str) -> np.ndarray:
    """P(n) for each day, shape (days, rounds). which = observed | predicted."""
    key = "observed" if which == "observed" else "predicted_at_measured_dephasing"
    offset = 0 if basis == "X" else len(ROUNDS)
    rows = []
    for day in DAYS:
        folder, k = folders(rep, distance, day)
        settings = load(RUNS / folder / "analysis.json")[key]["settings"]
        rows.append([settings[offset + j]["logical_error"][k] for j in range(len(ROUNDS))])
    return np.array(rows)


def label(ax, text: str) -> None:
    ax.text(-0.02, 1.02, text, transform=ax.transAxes, fontweight="bold", va="bottom", ha="right")


def fig_lambda() -> None:
    """Λ in X memory per rep, observed and predicted, grouped by placement."""
    combined = load(RUNS / "combined.json")
    reps = sorted(combined["reps"], key=lambda r: (r["rep"], r["day"]))
    obs = np.array([r["tests"]["X lambda"]["observed"] for r in reps])
    err = np.array([r["tests"]["X lambda"]["observed_uncertainty"] for r in reps])
    pred = np.array([r["tests"]["X lambda"]["predicted"] for r in reps])
    pooled = combined["pooled_x"]

    fig, ax = plt.subplots(figsize=(COLUMN, 3.0))
    half = len(reps) // 2
    # Placement 1 on top, then placement 2, each under its own heading; pooled at the bottom.
    y = np.concatenate([np.arange(half)[::-1] + half + 3, np.arange(half)[::-1] + 2]).astype(float)
    for i in range(len(reps)):
        ax.errorbar(obs[i], y[i], xerr=err[i], fmt="o", ms=3.6, color=D5, elinewidth=1.0, capsize=0)
        ax.plot(pred[i], y[i], marker="D", ms=3.4, mfc="white", mec=MUTED, mew=0.8, ls="none")
    for placement, top in ((1, y[0]), (2, y[half])):
        ax.text(0.31, top + 0.85, f"placement {placement}", ha="left", va="center", color=INK, fontsize=7, style="italic")
    ax.axvline(1.0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(0.985, y.max() + 0.85, r"$\Lambda=1$", ha="right", va="center", color=MUTED, fontsize=7)

    py = 0.0
    ax.errorbar(pooled["lambda"], py, xerr=pooled["lambda_uncertainty"], fmt="s", ms=4, color=INK, elinewidth=1.2)
    ax.plot(pooled["predicted"], py, marker="D", ms=3.4, mfc="white", mec=MUTED, mew=0.8, ls="none")
    ax.axhline(1.0, color=GRID, lw=0.6)

    labels = [f"{int(r['day'][-2:])} Oct" for r in reps] + ["all 10"]
    ax.set_yticks(list(y) + [py], labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.8, y.max() + 1.5)
    ax.set_xlim(0.3, 1.05)
    ax.set_xlabel(r"$\Lambda = \varepsilon_3/\varepsilon_5$, X memory")
    handles = [
        mpl.lines.Line2D([], [], marker="o", ms=3.6, color=D5, ls="none", label=r"observed, $\pm1\sigma$"),
        mpl.lines.Line2D([], [], marker="D", ms=3.4, mfc="white", mec=MUTED, ls="none", label="predicted"),
    ]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.1), handletextpad=0.3)
    fig.savefig(OUT / "lambda.pdf")
    plt.close(fig)


def fig_pn() -> None:
    """P(n) on each placement, X and Z memory, mean over the five days."""
    fig, axes = plt.subplots(2, 2, figsize=(PAGE * 0.72, 3.6), sharex=True, sharey=True)
    n = np.array(ROUNDS)
    for col, rep in enumerate((1, 2)):
        for row, basis in enumerate(("X", "Z")):
            ax = axes[row, col]
            for distance, colour in ((3, D3), (5, D5)):
                o = p_of_n(rep, distance, basis, "observed")
                p = p_of_n(rep, distance, basis, "predicted")
                ax.plot(n, p.mean(0), color=colour, lw=1.0, ls=(0, (4, 2)), alpha=0.9)
                ax.errorbar(
                    n, o.mean(0), yerr=o.std(0, ddof=1) / np.sqrt(len(DAYS)), fmt="o", ms=3.2,
                    color=colour, elinewidth=0.9, capsize=0, label=f"$d={distance}$",
                )
            ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (1, 2)))
            ax.set_ylim(0, 0.55)
            ax.set_xticks(ROUNDS)
            ax.grid(axis="y", color=GRID, lw=0.4)
            ax.set_title(f"{basis} memory, placement {rep}", loc="left", fontsize=8, color=INK)
            if col == 0:
                ax.set_ylabel(r"logical error $P(n)$")
            if row == 1:
                ax.set_xlabel(r"rounds $n$")
    handles = [
        mpl.lines.Line2D([], [], marker="o", ms=3.2, color=D3, ls="none", label="$d=3$ observed"),
        mpl.lines.Line2D([], [], marker="o", ms=3.2, color=D5, ls="none", label="$d=5$ observed"),
        mpl.lines.Line2D([], [], color=MUTED, ls=(0, (4, 2)), label="predicted"),
    ]
    axes[0, 0].legend(handles=handles, loc="lower right")
    for ax, tag in zip(axes.flat, "abcd"):
        label(ax, f"({tag})")
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    fig.savefig(OUT / "pn.pdf")
    plt.close(fig)


def fig_leakage() -> None:
    """(a) A flag that fired fires again: excess over the model by lag. (b) Leak rate by role."""
    repeats = load(DIAG / "repeats.json")
    why = load(DIAG / "why.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(PAGE * 0.72, 1.9))

    lags = sorted(repeats["lags"], key=int)
    t = np.array([int(k) for k in lags]) * ROUND_US
    chip = np.array([repeats["lags"][k]["chip"] for k in lags]) * 100
    model = np.array([repeats["lags"][k]["model"] for k in lags]) * 100
    a.plot(t, chip, "o-", color=D5, ms=3.2, label="chip")
    a.plot(t, model, "o-", color=MUTED, ms=3.2, mfc="white", label="calibration model")
    a.axhline(0, color=MUTED, lw=0.5)
    a.set_xlabel(r"time after a firing ($\mu$s)")
    a.set_ylabel("excess re-firing (points)")
    a.set_xlim(0, 52)
    a.legend(loc="upper right")
    label(a, "(a)")

    rng = np.random.default_rng(1)
    for role, colour, marker, name in (("relay", D3, "s", "relay"), ("z_flag", D5, "o", "flag")):
        rows = [r for r in why if r["role"] == role]
        x = np.array([r["cz_per_round"] for r in rows]) + rng.uniform(-0.35, 0.35, len(rows))
        y = np.array([r["leak_per_round"] for r in rows]) * 100
        b.plot(x, y, marker, ms=2.2, color=colour, alpha=0.45, mew=0, label=f"{name} ({len(rows)} series)")
        b.plot(np.mean([r["cz_per_round"] for r in rows]), np.median(y), marker, ms=5, color=colour, mec=INK, mew=0.6)
    b.set_ylabel("onset rate (% per round)")
    b.set_xlim(1.5, 9.8)
    b.set_xticks([3, 8.25], ["relays\n3 CZ, 1 readout, 1 reset", "flags\n8 CZ, 2 readouts, 2 resets"])
    b.tick_params(axis="x", length=0, labelsize=6.5)
    b.set_ylim(0, None)
    b.legend(loc="upper left", handletextpad=0.2)
    label(b, "(b)")
    fig.tight_layout(w_pad=1.6)
    fig.savefig(OUT / "leakage.pdf")
    plt.close(fig)


def fig_scale() -> None:
    """Λ on a uniform chip with every error rate scaled together."""
    data = load(DIAG / "uniform_scale.json")
    s = np.array(data["scales"])
    fig, ax = plt.subplots(figsize=(COLUMN, 1.9))
    for basis, colour, marker in (("X", D5, "o"), ("Z", D3, "s")):
        lam = np.array(data[basis]["lambda"])
        ax.plot(s, lam, marker + "-", color=colour, ms=3.2, label=f"{basis} memory")
        cross = data[basis]["crosses_at"]
        ax.plot([cross], [1.0], marker="|", ms=7, color=colour, mew=1.2)
        ax.annotate(f"{cross:.2f}", (cross, 1.0), xytext=(0, 6), textcoords="offset points", ha="center", color=colour, fontsize=7)
    ax.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.axvline(1.0, color=GRID, lw=0.6)
    ax.set_xlabel("error rates relative to Fez's 8 October medians")
    ax.set_ylabel(r"$\Lambda$")
    ax.set_xlim(0.15, 1.05)
    ax.legend(loc="upper right")
    fig.savefig(OUT / "scale.pdf")
    plt.close(fig)


def fig_layout() -> None:
    """The chip figure, drawn by the repository's own script from day 1's calibration."""
    calibration = RUNS / "ibm_fez-2026-10-04-r1" / "calibration.json"
    env_path = str(ROOT / "src")
    subprocess.run(
        [sys.executable, "figure1.py", "--calibration", str(calibration), "--out", str(OUT / "layout.pdf")],
        cwd=ROOT / "docs" / "figures",
        check=True,
        env={"PYTHONPATH": env_path, "PATH": "/usr/bin:/bin"},
    )


if __name__ == "__main__":
    fig_lambda()
    fig_pn()
    fig_leakage()
    fig_scale()
    fig_layout()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))
