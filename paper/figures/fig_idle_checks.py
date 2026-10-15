"""Joint-idle residuals and the raw-IQ association, from saved records only."""

import matplotlib.pyplot as plt
import numpy as np

from make_figures import D3, D5, GRID, INK, MUTED, PAGE, load


def make(results, output):
    checks = load(results / "idle_checks.json")["joint_residuals"]
    raw = load(results / "kingston_iq.json")
    fig, axes = plt.subplots(1, 3, figsize=(PAGE, 2.1), layout="constrained")
    ordered = sorted(checks, key=lambda q: checks[q]["chi2_red"])
    axes[0].plot(range(1, 26), [checks[q]["chi2_red"] for q in ordered], "o", color=INK, ms=2.5)
    axes[0].axhline(1, color=MUTED, ls="--", lw=0.7)
    axes[0].set(xlabel=r"qubits, ordered by $\chi^2/\nu$", ylabel=r"nominal $\chi^2/\nu$")
    axes[0].set_title("(a) Common idle fit", loc="left", fontsize=8)
    for q, color in ((ordered[12], D5), (ordered[-1], D3)):
        for basis, marker, ls in (("XY", "o", "-"), ("Z", "s", "--")):
            residual = checks[q]["views"][basis]["standardized_residual"]
            axes[1].plot(
                [1, 2, 4, 8, 16],
                residual,
                marker=marker,
                color=color,
                ls=ls,
                lw=0.7,
                ms=2.5,
                label=f"q{q}, {basis}",
            )
    axes[1].axhline(0, color=MUTED, lw=0.6)
    axes[1].set(xlabel="rounds", ylabel="standardized log residual", xticks=[1, 4, 8, 16])
    axes[1].set_title("(b) Median / worst-fitting qubits", loc="left", fontsize=8)
    axes[1].margins(y=0.15)
    references = {"R0": [], "R1": []}
    for arm, color in (("A", D5), ("B", D3)):
        hist = []
        for row in raw["qubits"]:
            projected = row["arms"][arm]["confident"]["histogram"]
            edges = np.asarray(projected["edges"])
            hist.append(projected["state"])
            for reference in references:
                references[reference].append(projected[reference])
        counts = np.sum(hist, axis=0)
        axes[2].stairs(
            counts / counts.sum() / np.diff(edges),
            edges,
            color=color,
            lw=0.9,
            label=f"state, arm {arm}",
        )
    for name, values in references.items():
        counts = np.sum(values, axis=0)
        axes[2].stairs(
            counts / counts.sum() / np.diff(edges),
            edges,
            color=MUTED if name == "R1" else INK,
            lw=0.7,
            ls="--" if name == "R1" else ":",
            label=rf"reference $|{name[-1]}\rangle$",
        )
    axes[2].set(xlabel=r"IQ along $|0\rangle$ to $|1\rangle$", ylabel="density", xlim=(-1, 4))
    axes[2].set_title("(c) Readouts in fitted state", loc="left", fontsize=8)
    axes[2].legend(fontsize=5.5, loc="upper right")
    for ax in axes:
        ax.grid(axis="y", color=GRID, lw=0.4)
    fig.savefig(output)
    plt.close(fig)
