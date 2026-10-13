"""Draw the paper's figures from the committed run folders and analysis results.

Run from the repository root:  python paper/figures/make_figures.py
Writes every figure of the paper as a vector PDF next to this file, gadget.pdf aside (drawn by
gadget.tex); the figures of the analyses after the run come from fig_*.py, which this calls.
Reads only runs/ and docs/analysis/results/.
The PDFs carry a fixed creation date (SOURCE_DATE_EPOCH), so a rerun writes the same bytes.
Needs the paper's font, Latin Modern: from the TeX installation, or from the folder that
LATIN_MODERN_DIR names (e.g. a MiKTeX fonts/opentype/public/lm).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[2]
RUNS = ROOT / "runs"
DIAG = ROOT / "docs" / "analysis" / "results"
OUT = Path(__file__).resolve().parent
# matplotlib and fontTools date the PDFs and their embedded fonts by it: 4 October 2026.
os.environ.setdefault("SOURCE_DATE_EPOCH", "1791072000")

DAYS = ["2026-10-04", "2026-10-05", "2026-10-06", "2026-10-07", "2026-10-08"]
ROUNDS = [1, 2, 3, 4, 6, 8]
ROUND_US = 7.9  # one round at both distances

# Two categorical slots, validated for colour-blind separation; d = 5 is the subject.
D3, D5 = "#eb6834", "#2a78d6"
# Qubit roles in Fig. 1's colours (docs/figures/draw_blueprint.py): X ancillas and flags, Z
# ancillas, relays. Detector classes take the colour of the qubits that measure them.
X_ROLE, Z_ROLE, RELAY_ROLE = "#E4574C", "#1F4E99", "#8C8C8C"
# Fig. 1 rings the qubits of placement 1 that were weak on every day (Sec. III).
WEAK_QUBITS = (83, 102, 149, 150)
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
COLUMN, PAGE = 3.375, 7.0  # revtex column and page widths, inches


def register_latin_modern() -> None:
    """Use the paper's font, Latin Modern: installed, from the TeX installation (kpsewhich), or
    from the folder LATIN_MODERN_DIR names. Stop rather than draw in another font."""
    from matplotlib import font_manager

    if any(f.name == "Latin Modern Roman" for f in font_manager.fontManager.ttflist):
        return
    folder = os.environ.get("LATIN_MODERN_DIR", "")
    if not folder:
        try:
            found = subprocess.run(
                ["kpsewhich", "lmroman10-regular.otf"], capture_output=True, text=True
            ).stdout.strip()
        except OSError:
            found = ""
        folder = str(Path(found).parent) if found else ""
    fonts = sorted(Path(folder).glob("lmroman*.otf")) if folder else []
    if not fonts:
        sys.exit(
            "Latin Modern not found: install TeX Live or the lmodern OTF fonts, "
            "or set LATIN_MODERN_DIR to the folder holding lmroman10-regular.otf"
        )
    for font in fonts:
        font_manager.fontManager.addfont(str(font))


register_latin_modern()
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


def binomial_curve(rep: int, distance: int, basis: str, n: np.ndarray) -> np.ndarray:
    """The binomial fit (1 - A (1 - 2 eps)^n) / 2 of each day, averaged over the days."""
    curves = load(DIAG / "fits.json")["curves"]
    days = []
    for day in DAYS:
        fit = curves[f"{day} r{rep} {basis} d{distance}"]["binomial"]
        days.append((1 - fit["A"] * (1 - 2 * fit["eps"]) ** n) / 2)
    return np.mean(days, axis=0)


def fig_pn() -> None:
    """P(n) of each pair, X and Z memory, mean over the five days: device, its binomial fit and
    the prediction."""
    fig, axes = plt.subplots(2, 2, figsize=(PAGE, 3.0), sharex=True, sharey=True)
    n = np.array(ROUNDS)
    smooth = np.linspace(1, 8, 57)
    for col, rep in enumerate((1, 2)):
        for row, basis in enumerate(("X", "Z")):
            ax = axes[row, col]
            # d = 3 and d = 5 differ in marker and dash as well as colour, so they survive greyscale.
            for distance, colour, marker, dash in (
                (3, D3, "s", (0, (5, 1.5, 1, 1.5))),
                (5, D5, "o", (0, (4, 2))),
            ):
                o = p_of_n(rep, distance, basis, "observed")
                p = p_of_n(rep, distance, basis, "predicted")
                ax.plot(n, p.mean(0), color=colour, lw=1.0, ls=dash, alpha=0.9)
                ax.plot(smooth, binomial_curve(rep, distance, basis, smooth), color=colour, lw=0.6)
                ax.errorbar(
                    n,
                    o.mean(0),
                    yerr=o.std(0, ddof=1) / np.sqrt(len(DAYS)),
                    fmt=marker,
                    ms=3.2,
                    color=colour,
                    elinewidth=0.9,
                    capsize=0,
                    label=f"$d={distance}$",
                )
            ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
            ax.set_ylim(0, 0.55)
            ax.set_xticks(ROUNDS)
            ax.grid(axis="y", color=GRID, lw=0.4)
            ax.set_title(f"{basis} memory, pair {rep}", loc="left", fontsize=8, color=INK)
            if col == 0:
                ax.set_ylabel(r"logical error $P_\mathrm{L}(n)$")
            if row == 1:
                ax.set_xlabel(r"rounds $n$")
    axes[0, 1].text(
        8.1,
        0.515,
        r"$P_\mathrm{L} = 1/2$ (random)",
        ha="right",
        va="bottom",
        fontsize=7,
        color=MUTED,
    )
    handles = [
        Line2D([], [], marker="s", ms=3.2, color=D3, ls="none", label="$d=3$"),
        Line2D([], [], marker="o", ms=3.2, color=D5, ls="none", label="$d=5$"),
        Line2D([], [], color=MUTED, lw=0.6, label="binomial fit"),
        Line2D([], [], color=MUTED, lw=1.0, ls=(0, (4, 2)), label="predicted"),
    ]
    axes[0, 0].legend(handles=handles, loc="lower right", ncol=2, columnspacing=1.0)
    for ax, tag in zip(axes.flat, "abcd"):
        label(ax, f"({tag})")
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    fig.savefig(OUT / "pn.pdf")
    plt.close(fig)


def fig_layout() -> None:
    """The chip figure, drawn by the repository's own script from day 1's calibration."""
    calibration = RUNS / "ibm_fez-2026-10-04-r1" / "calibration.json"
    env_path = str(ROOT / "src")
    # The paper's font, as in the other figures, embedded as TrueType rather than Type 3, as
    # arXiv and journals prefer. Importing this module registers Latin Modern; the rest of its
    # style is for plots, so it is reset before the chip figure is drawn.
    driver = (
        "import matplotlib, runpy, sys; sys.path.insert(0, sys.argv[3]); import make_figures; "
        "matplotlib.rcdefaults(); matplotlib.rcParams.update({'pdf.fonttype': 42, "
        "'font.family': 'serif', 'font.serif': ['Latin Modern Roman', 'CMU Serif'], "
        "'mathtext.fontset': 'cm'}); "
        "sys.argv = ['figure1.py', '--calibration', sys.argv[1], '--out', sys.argv[2], "
        "'--weak', sys.argv[4]]; "
        "runpy.run_path('figure1.py', run_name='__main__')"
    )
    subprocess.run(
        [
            sys.executable,
            "-c",
            driver,
            str(calibration),
            str(OUT / "layout.pdf"),
            str(OUT),
            ",".join(map(str, WEAK_QUBITS)),
        ],
        cwd=ROOT / "docs" / "figures",
        check=True,
        env={**os.environ, "PYTHONPATH": env_path},
    )


def fig_analyses() -> None:
    """The figures of the analyses after the run, each drawn by its own script."""
    import fig_B_detector_diagnostics
    import fig_C_idle_dephasing
    import fig_timing

    fig_timing.make(DIAG, OUT / "timing.pdf")
    fig_B_detector_diagnostics.make(DIAG, OUT / "B_detector_diagnostics.pdf")
    fig_C_idle_dephasing.make(DIAG, OUT / "C_idle_dephasing.pdf")


if __name__ == "__main__":
    fig_pn()
    fig_layout()
    fig_analyses()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))
