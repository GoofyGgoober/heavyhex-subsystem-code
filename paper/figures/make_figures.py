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


def fig_lambda() -> None:
    """Λ in X memory per rep, observed and predicted, grouped by placement."""
    combined = load(RUNS / "combined.json")
    reps = sorted(combined["reps"], key=lambda r: (r["rep"], r["day"]))
    obs = np.array([r["tests"]["X lambda"]["observed"] for r in reps])
    err = np.array([r["tests"]["X lambda"]["observed_uncertainty"] for r in reps])
    pred = np.array([r["tests"]["X lambda"]["predicted"] for r in reps])
    # The predictions frozen before each job, at f = 0 and f = 1 (effect_sizes.py).
    frozen = {
        (p["placement"], p["day"]): p
        for p in load(DIAG / "effect_sizes.json")["prejob"]["wls"]["per_repetition"]
    }
    bracket = np.array(
        [
            sorted(
                frozen[(r["rep"], DAYS.index(r["day"]) + 1)][f"X lambda frozen_f{f}"]
                for f in ("0.0", "1.0")
            )
            for r in reps
        ]
    )

    # Full column width with constrained layout, so LaTeX prints it at about 1:1 (fonts stay 7-8 pt)
    # and the figure is 2.0 in tall. The prespecified ten-repetition pool is not drawn: it treats
    # the repetitions as independent, and the paper quotes it once in the text.
    fig, ax = plt.subplots(figsize=(COLUMN, 2.0), layout="constrained")
    half = len(reps) // 2
    # Placement 1 on top, then placement 2, each under its own heading.
    y = np.concatenate([np.arange(half)[::-1] + half + 1, np.arange(half)[::-1]]).astype(float)
    for i in range(len(reps)):
        low = y[i] - 0.3  # the predictions just under the observation
        ax.plot(bracket[i], [low, low], color=MUTED, lw=2.4, alpha=0.4, solid_capstyle="butt")
        ax.errorbar(obs[i], y[i], xerr=err[i], fmt="o", ms=3.6, color=D5, elinewidth=1.0, capsize=0)
        ax.plot(pred[i], low, marker="D", ms=3.4, mfc="white", mec=MUTED, mew=0.8, ls="none")
    for placement, top in ((1, y[0]), (2, y[half])):
        ax.text(
            0.31,
            top + 0.85,
            f"placement {placement}",
            ha="left",
            va="center",
            color=INK,
            fontsize=7,
            style="italic",
        )
    ax.axvline(1.0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(0.985, y.max() + 0.85, r"$\Lambda=1$", ha="right", va="center", color=MUTED, fontsize=7)

    labels = [f"day {DAYS.index(r['day']) + 1}" for r in reps]
    ax.set_yticks(list(y), labels)
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.8, y.max() + 1.5)
    ax.set_xlim(0.3, 1.05)
    ax.set_xlabel(r"$\Lambda = \varepsilon_3/\varepsilon_5$, X memory")
    handles = [
        Line2D([], [], marker="o", ms=3.6, color=D5, ls="none", label=r"observed, $\pm1\sigma$"),
        Line2D([], [], marker="D", ms=3.4, mfc="white", mec=MUTED, ls="none", label="predicted"),
        Line2D([], [], color=MUTED, lw=2.4, alpha=0.4, label=r"before the job, $0 \leq f \leq 1$"),
    ]
    # Beside placement 2's points and left of the Lambda = 1 line, which the legend used to cross.
    ax.legend(
        handles=handles,
        loc="lower right",
        bbox_to_anchor=(0.92, 0.15),
        handletextpad=0.3,
        borderaxespad=0.0,
    )
    fig.savefig(OUT / "lambda.pdf")
    plt.close(fig)


def fig_pn() -> None:
    """P(n) on each placement, X and Z memory, mean over the five days."""
    fig, axes = plt.subplots(2, 2, figsize=(PAGE, 3.4), sharex=True, sharey=True)
    n = np.array(ROUNDS)
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
            ax.axhline(0.5, color=GRID, lw=1.4, zorder=0)
            ax.set_ylim(0, 0.55)
            ax.set_xticks(ROUNDS)
            ax.grid(axis="y", color=GRID, lw=0.4)
            ax.set_title(f"{basis} memory, placement {rep}", loc="left", fontsize=8, color=INK)
            if col == 0:
                ax.set_ylabel(r"logical error $P_\mathrm{L}(n)$")
            if row == 1:
                ax.set_xlabel(r"rounds $n$")
    handles = [
        Line2D([], [], marker="s", ms=3.2, color=D3, ls="none", label="$d=3$ observed"),
        Line2D([], [], marker="o", ms=3.2, color=D5, ls="none", label="$d=5$ observed"),
        Line2D([], [], color=D3, lw=1.0, ls=(0, (5, 1.5, 1, 1.5)), label="$d=3$ predicted"),
        Line2D([], [], color=D5, lw=1.0, ls=(0, (4, 2)), label="$d=5$ predicted"),
    ]
    axes[0, 0].legend(handles=handles, loc="lower right")
    for ax, tag in zip(axes.flat, "abcd"):
        label(ax, f"({tag})")
    fig.tight_layout(w_pad=1.0, h_pad=0.8)
    fig.savefig(OUT / "pn.pdf")
    plt.close(fig)


def fig_leakage() -> None:
    """(a) The lag-l persistence of the flag and relay readouts in both bases. (b) Onset rate by
    role."""
    from fig_D_flag_records import delta_panel

    causes = load(DIAG / "hmm_causes.json")
    fig, (a, b) = plt.subplots(1, 2, figsize=(PAGE, 2.1))
    delta_panel(a)
    label(a, "(a)")

    rng = np.random.default_rng(1)
    for role, colour, marker, name in (("relay", D3, "s", "relay"), ("z_flag", D5, "o", "flag")):
        rows = [r for r in causes if r["role"] == role]
        x = np.array([r["cz_per_round"] for r in rows]) + rng.uniform(-0.35, 0.35, len(rows))
        y = np.array([r["leak_per_round"] for r in rows]) * 100
        b.plot(
            x,
            y,
            marker,
            ms=2.2,
            color=colour,
            alpha=0.45,
            mew=0,
            label=f"{name} ({len(rows)} series)",
        )
        b.plot(
            np.mean([r["cz_per_round"] for r in rows]),
            np.median(y),
            marker,
            ms=5,
            color=colour,
            mec=INK,
            mew=0.6,
        )
    b.set_ylabel("onset rate (% per round)")
    b.set_xlim(1.5, 9.8)
    b.set_xticks([3, 8], ["relays\n3 CZ, 1 readout, 1 reset", "flags\n8 CZ, 2 readouts, 2 resets"])
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
        ax.annotate(
            f"{cross:.2f}",
            (cross, 1.0),
            xytext=(3, 5),
            textcoords="offset points",
            ha="left",
            color=colour,
            fontsize=7,
        )
    ax.axhline(1.0, color=MUTED, lw=0.6, ls=(0, (3, 2)))
    ax.set_xlabel("error rates relative to Fez's day-5 medians")
    ax.set_ylabel(r"$\Lambda$")
    ax.set_xlim(0.15, 1.05)
    ax.legend(loc="upper right")
    fig.savefig(OUT / "scale.pdf")
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
        "sys.argv = ['figure1.py', '--calibration', sys.argv[1], '--out', sys.argv[2]]; "
        "runpy.run_path('figure1.py', run_name='__main__')"
    )
    subprocess.run(
        [sys.executable, "-c", driver, str(calibration), str(OUT / "layout.pdf"), str(OUT)],
        cwd=ROOT / "docs" / "figures",
        check=True,
        env={**os.environ, "PYTHONPATH": env_path},
    )


def fig_analyses() -> None:
    """The figures of the analyses after the run, each drawn by its own script."""
    import fig_A_effect_sizes
    import fig_B_detector_diagnostics
    import fig_C_idle_dephasing
    import fig_D_flag_records

    fig_A_effect_sizes.make(DIAG, OUT / "A_effect_sizes.pdf")
    fig_B_detector_diagnostics.make(DIAG, OUT / "B_detector_diagnostics.pdf")
    fig_B_detector_diagnostics.make_pij(DIAG, OUT / "B_detector_diagnostics_pij.pdf")
    fig_C_idle_dephasing.make(DIAG, OUT / "C_idle_dephasing.pdf")
    fig_D_flag_records.make_trace(DIAG, OUT / "hmm_trace.pdf")


if __name__ == "__main__":
    fig_lambda()
    fig_pn()
    fig_leakage()
    fig_scale()
    fig_layout()
    fig_analyses()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))
