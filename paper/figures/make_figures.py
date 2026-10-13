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
from matplotlib.patches import Patch

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
    """The prespecified Λ_X test per pair-day: (a) observed and predicted with their own errors,
    and the prespecified pool; (b) z under both choices of σ_f."""
    combined = load(RUNS / "combined.json")
    reps = sorted(combined["reps"], key=lambda r: (r["rep"], r["day"]))
    tests = [r["tests"]["X lambda"] for r in reps]
    obs = np.array([t["observed"] for t in tests])
    s_obs = np.array([t["observed_uncertainty"] for t in tests])
    pred = np.array([t["predicted"] for t in tests])
    # sigma_M as the test uses it: simulator noise and the f term (the calibration term is 0).
    s_pred = np.array(
        [
            np.sqrt(t["simulator_noise"] ** 2 + t["from_f"] ** 2 + t["calibration_gap"] ** 2 / 2)
            for t in tests
        ]
    )
    s_test = np.hypot(s_obs, s_pred)
    assert np.allclose(2 * s_test, [t["bound"] for t in tests], rtol=1e-9)
    # z with sigma_f as prespecified and from the idle test's noise alone (prespecified_tests.py)
    tested = load(DIAG / "prespecified_tests.json")
    by_pair_day = {(r["pair"], r["day"]): r["wls"]["X lambda"] for r in tested["per_pair_day"]}
    days = [DAYS.index(r["day"]) + 1 for r in reps]
    z_pre = np.array([by_pair_day[(r["rep"], d)]["z"] for r, d in zip(reps, days)])
    z_noise = np.array(
        [by_pair_day[(r["rep"], d)]["z_idle_noise_only"] for r, d in zip(reps, days)]
    )
    assert np.allclose(z_pre, (obs - pred) / s_test, atol=1e-9)
    pool, pool_se, _, _, _, pool_pred = tested["headline"]["pool"][
        "Lambda_X, se, chi2_0/nu, t_0.95, upper bound, predicted"
    ]
    # Pair 1's patches share one job and one f, so its predictions stored before the job at f = 0
    # and f = 1 bracket the prediction; pair 2's patches ran in two jobs at two f, and are not
    # bracketed that way (effect_sizes.py, prejob).
    frozen = {
        (p["placement"], p["day"]): p
        for p in load(DIAG / "effect_sizes.json")["prejob"]["wls"]["per_repetition"]
    }

    # Full column width with constrained layout, so LaTeX prints it at about 1:1 (fonts 7-8 pt).
    fig, (ax, res) = plt.subplots(
        1, 2, figsize=(COLUMN, 2.75), sharey=True, width_ratios=[1.45, 1], layout="constrained"
    )
    half = len(reps) // 2
    # Pair 1 on top, then pair 2, each under its own heading.
    y = np.concatenate([np.arange(half)[::-1] + half + 1.4, np.arange(half)[::-1]]).astype(float)
    ax.axvspan(pool - pool_se, pool + pool_se, color=D5, alpha=0.13, lw=0, zorder=0)
    ax.axvline(pool_pred, color=MUTED, lw=0.8, ls=(0, (1, 1.5)), zorder=0)
    for i, r in enumerate(reps):
        low = y[i] - 0.3  # the prediction just under the observation
        if r["rep"] == 1:
            bracket = sorted(frozen[(1, days[i])][f"X lambda frozen_f{f}"] for f in ("0.0", "1.0"))
            ax.plot(bracket, [low, low], color=MUTED, lw=2.6, alpha=0.35, solid_capstyle="butt")
        ax.errorbar(
            pred[i],
            low,
            xerr=s_pred[i],
            fmt="D",
            ms=3.2,
            mfc="white",
            mec=MUTED,
            mew=0.8,
            ecolor=MUTED,
            elinewidth=0.9,
            capsize=0,
        )
        ax.errorbar(
            obs[i], y[i], xerr=s_obs[i], fmt="o", ms=3.4, color=D5, elinewidth=1.0, capsize=0
        )
        res.plot(z_pre[i], y[i], "o", ms=3.4, color=D5)
        res.plot(z_noise[i], y[i] - 0.3, "o", ms=3.4, mfc="white", mec=D5, mew=0.9)
    for pair, top, note in (
        (1, y[0], "one job, same qubits daily"),
        (2, y[half], "two jobs a day"),
    ):
        ax.text(
            0.31,
            top + 0.95,
            f"pair {pair}: {note}",
            ha="left",
            va="center",
            color=INK,
            fontsize=7,
            style="italic",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
            zorder=3,
        )
    ax.axvline(1.0, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    ax.text(0.985, -0.45, r"$\Lambda=1$", ha="right", va="center", color=MUTED, fontsize=7)
    for edge in (-2, 2):
        res.axvline(edge, color=MUTED, lw=0.7, ls=(0, (3, 2)))
    res.axvline(0.0, color=MUTED, lw=0.5)
    ax.set_yticks(list(y), [f"day {d}" for d in days])
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.8, y.max() + 1.6)
    ax.set_xlim(0.3, 1.05)
    ax.set_xlabel(r"$\Lambda_\mathrm{X} = \varepsilon_3/\varepsilon_5$")
    res.set_xlim(-4.2, 4.2)
    res.set_xticks([-4, -2, 0, 2, 4])
    res.set_xlabel(r"$z$ of $\Lambda_\mathrm{X}$")
    res.tick_params(axis="y", length=0)
    label(ax, "(a)")
    label(res, "(b)")
    handles = [
        Line2D([], [], marker="o", ms=3.4, color=D5, ls="none", label=r"observed, $\pm\sigma_O$"),
        Line2D(
            [],
            [],
            marker="D",
            ms=3.2,
            mfc="white",
            mec=MUTED,
            color=MUTED,
            lw=0.9,
            label=r"predicted, $\pm\sigma_M$",
        ),
        Line2D([], [], color=MUTED, lw=2.6, alpha=0.35, label=r"before the job, $0\leq f\leq 1$"),
        Patch(color=D5, alpha=0.13, lw=0, label=r"pool, $\pm\sigma$"),
        Line2D([], [], color=MUTED, lw=0.8, ls=(0, (1, 1.5)), label="pooled prediction"),
        Line2D(
            [],
            [],
            marker="o",
            ms=3.4,
            color=D5,
            mfc="white",
            mew=0.9,
            ls="none",
            label=r"(b) $\sigma_f$ from idle noise only",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="outside lower center",
        ncol=2,
        handletextpad=0.3,
        columnspacing=0.8,
        fontsize=6.5,
    )
    fig.savefig(OUT / "lambda.pdf")
    plt.close(fig)


def fig_pn() -> None:
    """P(n) of each pair, X and Z memory, mean over the five days."""
    fig, axes = plt.subplots(2, 2, figsize=(PAGE, 3.0), sharex=True, sharey=True)
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
            # The log-linear fit leaves out points with P >= 1/2.
            ax.axhline(0.5, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=0)
            ax.set_ylim(0, 0.55)
            ax.set_xticks(ROUNDS)
            ax.grid(axis="y", color=GRID, lw=0.4)
            ax.set_title(f"{basis} memory, pair {rep}", loc="left", fontsize=8, color=INK)
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
    import fig_timing

    fig_timing.make(DIAG, OUT / "timing.pdf")
    fig_A_effect_sizes.make(DIAG, OUT / "A_effect_sizes.pdf")
    fig_B_detector_diagnostics.make(DIAG, OUT / "B_detector_diagnostics.pdf")
    fig_B_detector_diagnostics.make_pij(DIAG, OUT / "B_detector_diagnostics_pij.pdf")
    fig_C_idle_dephasing.make(DIAG, OUT / "C_idle_dephasing.pdf")
    fig_D_flag_records.make(DIAG, OUT / "flag_records.pdf")


if __name__ == "__main__":
    fig_lambda()
    fig_pn()
    fig_scale()
    fig_layout()
    fig_analyses()
    print("wrote", ", ".join(sorted(p.name for p in OUT.glob("*.pdf"))))
