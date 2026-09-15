"""Disjoint d=3 and d=5 heavy-hex layouts on the cached ibm_fez map.

Operators follow Sundaresan et al., Nat. Commun. 14, 2852 (2023), eqs (1)-(4):
code qubits Q1-Q9 are labelled column-major, so the X gauges are the six
within-row pairs and the Z gauges are the two corner column pairs plus two
2x2 blocks.

The 23-qubit device patch is the Falcon-27 map minus its four degree-1
corners (QF0, QF6, QF20, QF26), carried over to ibm_fez by d3_fez_layout.json.
Roles are reconstructed from the paper's constraints and checked against every
independent number it reports: 9 code qubits; 4 Z-gauge syndromes on the four
vertical links; 6 in-row flags measuring the X gauges, four of which double as
flags in the weight-4 Z circuits; 4 relays whose flag outcomes are discarded;
12 measurements in a Z round + 6 in an X round = 18 per round.

Map data: coupling map + grid coords previously cached to fez_map.json.
Regeneration is entirely offline and never contacts IBM services.

The d=3 placement is retained exactly. The d=5 extension below it uses the
same two-relay boundary arms: 25 data + 20 X ancillas + 12 Z ancillas + 8
relays = 65 sites. These are coupling layouts, not timed executable circuits.
"""

from __future__ import annotations

import json
import os
import re

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from heavyhex_layout import build_layout
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

HERE = os.path.dirname(os.path.abspath(__file__))
GOLD = "#F2C230"
GREEN = "#1B7A3D"
X_RED = "#E4574C"
Z_BLUE = "#2A78D6"
GREY = "#9A9A9A"
INK = "#333333"
TINT = {"#2A78D6": "#CDDFF5", "#E4574C": "#F7D4D1"}

# --- code definition (column-major labels: Qk sits at row (k-1)%3+1) --------
CELL = {k: ((k - 1) % 3 + 1, (k - 1) // 3 + 1) for k in range(1, 10)}
X_GAUGE = {
    "X1X4": (1, 4),
    "X2X5": (2, 5),
    "X3X6": (3, 6),
    "X4X7": (4, 7),
    "X5X8": (5, 8),
    "X6X9": (6, 9),
}
Z_GAUGE = {"Z1Z2": (1, 2), "Z8Z9": (8, 9), "Z2Z3Z5Z6": (2, 3, 5, 6), "Z4Z5Z7Z8": (4, 5, 7, 8)}
X_STAB = {"X1X2X4X5": (1, 2, 4, 5), "X4X7": (4, 7), "X3X6": (3, 6), "X5X6X8X9": (5, 6, 8, 9)}
Z_STAB = {"Z1Z2Z4Z5Z7Z8": (1, 2, 4, 5, 7, 8), "Z2Z3Z5Z6Z8Z9": (2, 3, 5, 6, 8, 9)}
X_L, Z_L = (1, 2, 3), (1, 4, 7)

# --- device assignment, in Falcon-27 numbering ------------------------------
DATA = {1: 2, 2: 10, 3: 17, 4: 5, 5: 13, 6: 21, 7: 9, 8: 16, 9: 24}
X_ANC = {"X1X4": 3, "X2X5": 12, "X3X6": 18, "X4X7": 8, "X5X8": 14, "X6X9": 23}
Z_ANC = {"Z1Z2": 4, "Z4Z5Z7Z8": 11, "Z2Z3Z5Z6": 15, "Z8Z9": 22}
# Z-gauge CX chains: the weight-2 gauges reach their data through relays, the
# weight-4 gauges through in-row flags that each pick up two data qubits.
Z_ARMS = {
    "Z1Z2": [[2, 1, 4], [10, 7, 4]],
    "Z8Z9": [[16, 19, 22], [24, 25, 22]],
    "Z4Z5Z7Z8": [[8, 11], [14, 11]],
    "Z2Z3Z5Z6": [[12, 15], [18, 15]],
}
RELAY = [1, 7, 19, 25]
FLAG_RULE = {12: 2, 18: 6, 8: 4, 14: 8}  # flag QF -> Z correction on code qubit


def sub(name: str) -> str:
    """X1X4 -> mathtext X_1X_4."""
    return "$" + re.sub(r"(\d+)", r"_{\1}", name) + "$"


def cell(k, d=3):
    r, c = (k - 1) % d + 1, (k - 1) // d + 1
    return c, -r


def main() -> None:
    with open(os.path.join(HERE, "fez_map.json")) as file:
        device_map = json.load(file)
    coords, edges = device_map["coords"], device_map["edges"]
    n = len(coords)
    bonds = {tuple(sorted(e)) for e in edges}
    pos = {q: (float(coords[q][0]), float(coords[q][1])) for q in range(n)}
    qf2fez = {
        int(k): v for k, v in json.load(open(os.path.join(HERE, "d3_fez_layout.json"))).items()
    }
    F = qf2fez.__getitem__

    role = {}
    for k, qf in DATA.items():
        role[F(qf)] = ("data", f"Q{k}")
    for g, qf in X_ANC.items():
        role[F(qf)] = ("xanc", g)
    for g, qf in Z_ANC.items():
        role[F(qf)] = ("zanc", g)
    for qf in RELAY:
        role[F(qf)] = ("relay", "")
    assert len(role) == 23, len(role)

    wires = []
    for g, (a, b) in X_GAUGE.items():
        wires += [("x", F(DATA[a]), F(X_ANC[g])), ("x", F(X_ANC[g]), F(DATA[b]))]
    for g, arms in Z_ARMS.items():
        for arm in arms:
            wires += [("z", F(u), F(v)) for u, v in zip(arm, arm[1:])]
    assert not [w for w in wires if tuple(sorted(w[1:])) not in bonds]
    print(f"{len(role)} qubits, {len(wires)} couplings, all real ibm_fez bonds")
    d5 = build_layout(5, coords, edges, first_data_position=(3, 7))
    if set(role) & d5.physical_qubits:
        raise ValueError("the two patches must not share any physical qubits")
    d5_role = {q: ("data", f"Q{k}") for k, q in d5.data.items()}
    d5_role.update({q: ("xanc", g) for g, q in d5.x_ancillas.items()})
    d5_role.update({q: ("zanc", g) for g, q in d5.z_ancillas.items()})
    d5_role.update({q: ("relay", "") for q in d5.relays})
    print(f"d=5: {len(d5_role)} qubits, {len(d5.wires)} real bonds; no shared sites")

    halo = [pe.withStroke(linewidth=2.6, foreground="white")]
    box = dict(facecolor="white", edgecolor="none", pad=0.12, alpha=0.85)
    boxz = dict(facecolor=TINT[Z_BLUE], edgecolor="none", pad=0.12)

    fig = plt.figure(figsize=(18, 20))
    gs = fig.add_gridspec(
        4,
        12,
        height_ratios=[1.22, 0.92, 1.40, 1.05],
        hspace=0.30,
        wspace=0.30,
        left=0.025,
        right=0.985,
        top=0.94,
        bottom=0.09,
    )
    ax_chip = fig.add_subplot(gs[0, 0:3])
    ax_dev = fig.add_subplot(gs[0, 3:8])
    ax_txt = fig.add_subplot(gs[0, 8:12])
    ax_xg = fig.add_subplot(gs[1, 0:3])
    ax_zg = fig.add_subplot(gs[1, 3:6])
    ax_xs = fig.add_subplot(gs[1, 6:9])
    ax_zs = fig.add_subplot(gs[1, 9:12])
    ax_d5 = fig.add_subplot(gs[2, 0:7])
    ax_d5_txt = fig.add_subplot(gs[2, 7:12])
    d5_grids = tuple(fig.add_subplot(gs[3, i : i + 3]) for i in range(0, 12, 3))
    grids = (ax_xg, ax_zg, ax_xs, ax_zs) + d5_grids
    for ax in (ax_chip, ax_dev, ax_txt, ax_d5, ax_d5_txt) + grids:
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
    for ax in (ax_chip, ax_dev, ax_d5) + grids:
        ax.set_aspect("equal")
        ax.set_anchor("N")

    xs = [pos[q][0] for q in role]
    ys = [pos[q][1] for q in role]

    # (a) the whole chip, patch highlighted ------------------------------
    for a, b in edges:
        ax_chip.plot(*zip(pos[a], pos[b]), color="#E1E1E1", lw=0.6, zorder=1)
    ax_chip.scatter(*zip(*pos.values()), s=5, color="#D8D8D8", zorder=2)
    for kind, u, v in wires:
        ax_chip.plot(*zip(pos[u], pos[v]), color=Z_BLUE if kind == "z" else X_RED, lw=1.5, zorder=3)
    for q, (r, _) in role.items():
        col = {"data": GOLD, "xanc": X_RED, "zanc": Z_BLUE, "relay": "white"}[r]
        ax_chip.scatter(*pos[q], s=20, color=col, edgecolor=INK, lw=0.5, zorder=4)
    ax_chip.add_patch(
        Rectangle(
            (min(xs) - 0.7, min(ys) - 0.7),
            max(xs) - min(xs) + 1.4,
            max(ys) - min(ys) + 1.4,
            facecolor="none",
            edgecolor=INK,
            lw=1.2,
            linestyle="--",
            zorder=5,
        )
    )
    ax_chip.set_xlim(0.4, 16.6)
    ax_chip.set_ylim(16.0, -0.4)  # Include the lower patch boundary below device row 15.
    for kind, u, v in d5.wires:
        ax_chip.plot(*zip(pos[u], pos[v]), color=Z_BLUE if kind == "z" else X_RED, lw=1.5, zorder=3)
    for q, (r, _) in d5_role.items():
        col = {"data": GOLD, "xanc": X_RED, "zanc": Z_BLUE, "relay": "white"}[r]
        ax_chip.scatter(*pos[q], s=20, color=col, edgecolor=INK, lw=0.5, zorder=4)
    d5_xs = [pos[q][0] for q in d5_role]
    d5_ys = [pos[q][1] for q in d5_role]
    ax_chip.add_patch(
        Rectangle(
            (min(d5_xs) - 0.7, min(d5_ys) - 0.7),
            max(d5_xs) - min(d5_xs) + 1.4,
            max(d5_ys) - min(d5_ys) + 1.4,
            facecolor="none",
            edgecolor=INK,
            lw=1.2,
            linestyle="--",
            zorder=5,
        )
    )
    for patch_role, label in ((role, "d=3\n23 sites"), (d5_role, "d=5\n65 sites")):
        patch_x = [pos[q][0] for q in patch_role]
        patch_y = [pos[q][1] for q in patch_role]
        ax_chip.text(
            max(patch_x) + 1.0,
            (min(patch_y) + max(patch_y)) / 2,
            label,
            fontsize=9,
            va="center",
            fontweight="bold",
            bbox=box,
        )
    ax_chip.set_title("(a) both patches: 88 of 156 sites; no shared qubits", fontsize=10)

    # (b) the circuit at true device coordinates -------------------------
    win = (min(xs) - 1.3, max(xs) + 1.3, min(ys) - 1.0, max(ys) + 1.35)

    def inwin(q):
        return win[0] <= pos[q][0] <= win[1] and win[2] <= pos[q][1] <= win[3]

    near = {
        w
        for q in role
        for w in range(n)
        if tuple(sorted((q, w))) in bonds and w not in role and inwin(w)
    }
    for a, b in edges:
        if {a, b} <= (set(role) | near):
            ax_dev.plot(*zip(pos[a], pos[b]), color="#E2E2E2", lw=1.4, zorder=1)
    ax_dev.scatter(
        [pos[q][0] for q in near], [pos[q][1] for q in near], s=30, color="#D8D8D8", zorder=2
    )
    for kind, u, v in wires:
        ax_dev.plot(
            *zip(pos[u], pos[v]),
            color=Z_BLUE if kind == "z" else X_RED,
            lw=2.8,
            solid_capstyle="round",
            zorder=3,
        )
    for q, (r, name) in role.items():
        x, y = pos[q]
        if r == "data":
            ax_dev.scatter([x], [y], s=170, color=GOLD, edgecolor=INK, lw=1.1, zorder=6)
            ax_dev.text(
                x,
                y - 0.22,
                f"${name[0]}_{{{name[1:]}}}$",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                zorder=7,
                path_effects=halo,
            )
        elif r == "relay":
            ax_dev.scatter([x], [y], s=120, color="white", edgecolor=GREY, lw=1.8, zorder=6)
        else:
            col = X_RED if r == "xanc" else Z_BLUE
            ax_dev.scatter(
                [x],
                [y],
                s=130,
                color=col,
                edgecolor="white",
                lw=1.0,
                marker="D" if r == "xanc" else "o",
                zorder=6,
            )
            dx, dy, ha, va = (
                (0.28, 0.28, "left", "top") if r == "xanc" else (-0.26, 0.0, "right", "center")
            )
            ax_dev.text(
                x + dx,
                y + dy,
                sub(name),
                ha=ha,
                va=va,
                fontsize=10,
                color=col,
                fontweight="bold",
                zorder=8,
                bbox=box,
            )
        ix, iy, iha, iva = {
            "data": (0.0, 0.17, "center", "top"),
            "xanc": (0.18, -0.17, "left", "bottom"),
            "zanc": (0.26, 0.0, "left", "center"),
            "relay": (0.15, -0.15, "left", "bottom"),
        }[r]
        ax_dev.text(
            x + ix,
            y + iy,
            f"{q}",
            ha=iha,
            va=iva,
            fontsize=7.5,
            color=GREY,
            zorder=7,
            path_effects=halo,
        )
    ax_dev.set_xlim(win[0], win[1])
    ax_dev.set_ylim(win[3], win[2])
    ax_dev.set_title("(b) d=3 coupling layout \u2014 every link is a real fez bond", fontsize=10)

    # (c) operator table --------------------------------------------------
    rows = [("head", "4 Z gauges measured (syndrome on a vertical link)", "")]
    rows += [("z", sub(g), f"q{F(Z_ANC[g])}") for g in ("Z1Z2", "Z2Z3Z5Z6", "Z4Z5Z7Z8", "Z8Z9")]
    rows += [("head", "6 X gauges measured (in-row flag qubit)", "")]
    rows += [("x", sub(g), f"q{F(X_ANC[g])}") for g in X_GAUGE]
    rows += [
        ("head", "2 Z stabilizers, inferred as gauge products", ""),
        ("z", sub("Z1Z2Z4Z5Z7Z8"), f"= {sub('Z1Z2')}\u00b7{sub('Z4Z5Z7Z8')}"),
        ("z", sub("Z2Z3Z5Z6Z8Z9"), f"= {sub('Z2Z3Z5Z6')}\u00b7{sub('Z8Z9')}"),
        ("head", "4 X stabilizers, inferred as gauge products", ""),
        ("x", sub("X1X2X4X5"), f"= {sub('X1X4')}\u00b7{sub('X2X5')}"),
        ("x", sub("X5X6X8X9"), f"= {sub('X5X8')}\u00b7{sub('X6X9')}"),
        ("x", sub("X4X7"), "= gauge, already central"),
        ("x", sub("X3X6"), "= gauge, already central"),
        ("head", "logical operators", ""),
        ("g", f"$X_L$ = {sub('X1X2X3')}", "column 1 of the grid"),
        ("g", f"$Z_L$ = {sub('Z1Z4Z7')}", "row 1 of the grid"),
    ]
    colour = {"z": Z_BLUE, "x": X_RED, "g": GREEN}
    y = 1.0
    for kind, left, right in rows:
        if kind == "head":
            y -= 0.012
            ax_txt.text(
                0.0,
                y,
                left,
                transform=ax_txt.transAxes,
                ha="left",
                va="top",
                fontsize=9.2,
                color=INK,
                fontweight="bold",
            )
            y -= 0.054
            continue
        ax_txt.text(
            0.035,
            y,
            left,
            transform=ax_txt.transAxes,
            ha="left",
            va="top",
            fontsize=9.8,
            color=colour[kind],
        )
        ax_txt.text(
            0.40,
            y,
            right,
            transform=ax_txt.transAxes,
            ha="left",
            va="top",
            fontsize=9.8,
            color=colour[kind],
        )
        y -= 0.0455
    ax_txt.set_title("(c) d=3 operators and ancilla assignments", fontsize=10)

    # (d)-(g) the code grid ----------------------------------------------
    def grid(ax, pad=0.62, d=3):
        for k in range(1, d * d + 1):
            x, y = cell(k, d)
            ax.scatter([x], [y], s=165, color=GOLD, edgecolor=INK, lw=1.1, zorder=6)
            ax.text(x, y - 0.015, f"{k}", ha="center", va="center", fontsize=8.5, zorder=7)
        ax.set_xlim(1 - pad, d + pad)
        ax.set_ylim(-d - pad, -1 + pad)

    def rect(ax, ks, color, *, fill, pad, lw=1.8, ls="-", z=3, d=3):
        pts = [cell(k, d) for k in ks]
        x0, y0 = min(p[0] for p in pts) - pad, min(p[1] for p in pts) - pad
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                max(p[0] for p in pts) - x0 + pad,
                max(p[1] for p in pts) - y0 + pad,
                boxstyle="round,pad=0.0,rounding_size=0.16",
                facecolor=TINT[color] if fill else "none",
                edgecolor=color,
                linewidth=lw,
                linestyle=ls,
                zorder=z,
            )
        )

    def bar(
        ax, ks, color, label, *, z=3, loff=(0.17, 0.0), lha="left", lva="center", lbox=None, d=3
    ):
        (x1, y1), (x2, y2) = cell(ks[0], d), cell(ks[1], d)
        ax.plot([x1, x2], [y1, y2], color=color, lw=7, alpha=0.35, solid_capstyle="round", zorder=z)
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.scatter(
            [mx],
            [my],
            s=115,
            color=color,
            edgecolor="white",
            lw=1.0,
            marker="D" if color == X_RED else "o",
            zorder=z + 3,
        )
        ax.text(
            mx + loff[0],
            my + loff[1],
            label,
            ha=lha,
            va=lva,
            fontsize=8.2,
            color=color,
            fontweight="bold",
            zorder=z + 4,
            bbox=lbox or box,
        )

    for g, ks in X_GAUGE.items():
        bar(ax_xg, ks, X_RED, f"q{F(X_ANC[g])}", loff=(0.0, 0.21), lha="center", lva="bottom")
    grid(ax_xg)
    ax_xg.set_title("(d) 6 X gauges measured, one flag qubit each", fontsize=10)

    for g, ks in Z_GAUGE.items():
        if len(ks) == 2:
            bar(ax_zg, ks, Z_BLUE, f"q{F(Z_ANC[g])}", z=4)
        else:
            filled = g == "Z4Z5Z7Z8"
            rect(
                ax_zg,
                ks,
                Z_BLUE,
                fill=filled,
                pad=0.30 if filled else 0.21,
                lw=1.8 if filled else 2.4,
                z=2 if filled else 5,
            )
            x = sum(cell(k)[0] for k in ks) / 4
            y = sum(cell(k)[1] for k in ks) / 4
            ax_zg.scatter([x], [y], s=115, color=Z_BLUE, edgecolor="white", lw=1.0, zorder=7)
            ax_zg.text(
                x + 0.17,
                y,
                f"q{F(Z_ANC[g])}",
                ha="left",
                va="center",
                fontsize=8.2,
                color=Z_BLUE,
                fontweight="bold",
                zorder=8,
                bbox=boxz if filled else box,
            )
    grid(ax_zg)
    ax_zg.set_title("(e) 4 Z gauges measured, one syndrome qubit each", fontsize=10)

    lab = {
        "X1X2X4X5": (0.9, -0.42, "left", "bottom"),
        "X4X7": (3.1, -0.42, "right", "bottom"),
        "X3X6": (0.9, -3.58, "left", "top"),
        "X5X6X8X9": (3.1, -3.58, "right", "top"),
    }
    for i, (g, ks) in enumerate(X_STAB.items()):
        rect(
            ax_xs,
            ks,
            X_RED,
            fill=False,
            pad=0.32 if len(ks) == 4 else 0.16,
            lw=1.9,
            ls=(0, (5, 2.5)) if i % 2 == 0 else (2.5, (5, 2.5)),
            z=4,
        )
        lx, ly, ha, va = lab[g]
        ax_xs.text(
            lx, ly, sub(g), color=X_RED, fontsize=8.6, fontweight="bold", ha=ha, va=va, zorder=7
        )
    grid(ax_xs, pad=0.75)
    ax_xs.set_title(
        "(f) 4 X stabilizers: two are gauge products, two are\ngauges that are already central",
        fontsize=10,
    )

    for i, (g, ks) in enumerate(Z_STAB.items()):
        rect(
            ax_zs,
            ks,
            Z_BLUE,
            fill=i == 0,
            pad=0.36 - 0.19 * i,
            lw=1.9,
            ls=(0, (5, 2.5)) if i == 0 else (2.5, (5, 2.5)),
            z=2 + i,
        )
    ax_zs.text(
        2.35,
        -1.5,
        sub("Z1Z2Z4Z5Z7Z8"),
        color=Z_BLUE,
        fontsize=8.6,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=7,
        bbox=boxz,
    )
    ax_zs.text(
        2.35,
        -2.5,
        sub("Z2Z3Z5Z6Z8Z9"),
        color=Z_BLUE,
        fontsize=8.6,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=7,
        bbox=box,
    )
    ax_zs.plot(
        [cell(k)[0] for k in X_L],
        [cell(k)[1] for k in X_L],
        color=GREEN,
        lw=3.4,
        solid_capstyle="round",
        zorder=5,
    )
    ax_zs.plot(
        [cell(k)[0] for k in Z_L],
        [cell(k)[1] for k in Z_L],
        color=GREEN,
        lw=3.4,
        ls=":",
        solid_capstyle="round",
        zorder=5,
    )
    grid(ax_zs, pad=0.75)
    ax_zs.text(
        1,
        -3.42,
        "$X_L$",
        color=GREEN,
        fontsize=11,
        fontweight="bold",
        ha="center",
        va="top",
        zorder=7,
        path_effects=halo,
    )
    ax_zs.text(
        3.3,
        -1,
        "$Z_L$",
        color=GREEN,
        fontsize=11,
        fontweight="bold",
        ha="left",
        va="center",
        zorder=7,
        path_effects=halo,
    )
    ax_zs.set_title(
        "(g) 2 Z stabilizers, both gauge products;\nlogical operators in green", fontsize=10
    )

    # (h) relay-preserving d=5 extension on the same physical chip.
    d5_window = (min(d5_xs) - 1.0, max(d5_xs) + 1.0, min(d5_ys) - 0.8, max(d5_ys) + 0.8)
    for a, b in edges:
        if all(
            d5_window[0] <= pos[q][0] <= d5_window[1] and d5_window[2] <= pos[q][1] <= d5_window[3]
            for q in (a, b)
        ):
            ax_d5.plot(*zip(pos[a], pos[b]), color="#E2E2E2", lw=1.2, zorder=1)
    for kind, u, v in d5.wires:
        ax_d5.plot(
            *zip(pos[u], pos[v]),
            color=Z_BLUE if kind == "z" else X_RED,
            lw=2.8,
            solid_capstyle="round",
            zorder=3,
        )
    for q, (r, name) in d5_role.items():
        x, y = pos[q]
        color = {"data": GOLD, "xanc": X_RED, "zanc": Z_BLUE, "relay": "white"}[r]
        ax_d5.scatter(
            x,
            y,
            s=135 if r == "data" else 95,
            color=color,
            edgecolor=INK if r == "data" else GREY if r == "relay" else "white",
            lw=1.1,
            marker="D" if r == "xanc" else "o",
            zorder=6,
        )
        if r == "data":
            ax_d5.text(
                x,
                y - 0.23,
                f"$Q_{{{name[1:]}}}$",
                ha="center",
                va="bottom",
                fontsize=10.5,
                zorder=7,
                path_effects=halo,
            )
        ax_d5.text(
            x,
            y + 0.2,
            f"q{q}",
            ha="center",
            va="top",
            fontsize=7.4,
            color=GREY if r in ("data", "relay") else color,
            path_effects=halo,
            zorder=7,
        )
    ax_d5.set_xlim(d5_window[:2])
    ax_d5.set_ylim(d5_window[3], d5_window[2])
    ax_d5.set_title(
        "(h) d=5 coupling layout: 25 data + 20 X ancillas + 12 Z ancillas + 8 relays = 65 sites",
        fontsize=10,
    )

    # (i) all measured d=5 gauges; stabilizers below are inferred products.
    for left, title, gauges, ancillas, color in (
        (0.0, "20 X gauges", d5.x_gauges, d5.x_ancillas, X_RED),
        (0.49, "12 Z gauges", d5.z_gauges, d5.z_ancillas, Z_BLUE),
    ):
        ax_d5_txt.text(
            left,
            1.0,
            title,
            transform=ax_d5_txt.transAxes,
            fontsize=10,
            fontweight="bold",
            color=INK,
            va="top",
        )
        for index, name in enumerate(gauges):
            y = 0.94 - index * 0.046
            ax_d5_txt.text(
                left,
                y,
                sub(name),
                transform=ax_d5_txt.transAxes,
                fontsize=9.6,
                color=color,
                va="top",
            )
            ax_d5_txt.text(
                left + 0.42,
                y,
                f"q{ancillas[name]}",
                transform=ax_d5_txt.transAxes,
                ha="right",
                va="top",
                fontsize=9.6,
                color=color,
            )
    ax_d5_txt.text(
        0.49,
        0.32,
        "Inferred: 12 X + 4 Z stabilizers\n"
        "One logical qubit; 8 gauge qubits\n"
        "Q labels restart within each patch.\n"
        "q labels are physical Fez IDs.",
        transform=ax_d5_txt.transAxes,
        va="top",
        fontsize=9.5,
        linespacing=1.6,
        color=INK,
    )
    ax_d5_txt.text(
        0.49,
        0.11,
        "$X_L$ = " + sub("X1X2X3X4X5") + "\n" + "$Z_L$ = " + sub("Z1Z6Z11Z16Z21"),
        transform=ax_d5_txt.transAxes,
        va="top",
        fontsize=10,
        linespacing=1.7,
        color=GREEN,
    )
    ax_d5_txt.set_title("(i) d=5 operators and ancilla assignments", fontsize=10)

    # (j)-(m) the same four views as the d=3 row, with identical conventions.
    d5_xg, d5_zg, d5_xs_ax, d5_zs_ax = d5_grids
    for name, support in d5.x_gauges.items():
        bar(
            d5_xg,
            support,
            X_RED,
            f"q{d5.x_ancillas[name]}",
            d=5,
            loff=(0.0, 0.23),
            lha="center",
            lva="bottom",
        )
    for name, support in d5.z_gauges.items():
        if len(support) == 2:
            bar(
                d5_zg,
                support,
                Z_BLUE,
                f"q{d5.z_ancillas[name]}",
                d=5,
                loff=(-0.20, 0.0) if support[0] <= 5 else (0.20, 0.0),
                lha="right" if support[0] <= 5 else "left",
            )
        else:
            rect(d5_zg, support, Z_BLUE, fill=True, pad=0.15, d=5)
            x = sum(cell(k, 5)[0] for k in support) / 4
            y = sum(cell(k, 5)[1] for k in support) / 4
            d5_zg.text(
                x,
                y,
                f"q{d5.z_ancillas[name]}",
                color=Z_BLUE,
                fontsize=8.2,
                ha="center",
                va="center",
                bbox=boxz,
                zorder=7,
            )
    for support in d5.x_stabilizers.values():
        rect(
            d5_xs_ax,
            support,
            X_RED,
            fill=False,
            pad=0.19 if len(support) == 4 else 0.13,
            ls="--",
            d=5,
        )
    for index, support in enumerate(d5.z_stabilizers.values()):
        rect(
            d5_zs_ax,
            support,
            Z_BLUE,
            fill=False,
            pad=0.30 - 0.10 * (index % 2),
            ls=(0 if index % 2 == 0 else 2.5, (5, 2.5)),
            d=5,
        )
        d5_zs_ax.text(
            5.35,
            -index - 1.5,
            f"$S^Z_{{{index + 1}}}$",
            color=Z_BLUE,
            fontsize=9,
            ha="left",
            va="center",
            zorder=7,
        )
    d5_zs_ax.plot([1, 1], [-1, -5], color=GREEN, lw=3.4, zorder=5)
    d5_zs_ax.plot([1, 5], [-1, -1], color=GREEN, lw=3.4, ls=":", zorder=5)
    d5_zs_ax.text(
        1, -5.42, "$X_L$", color=GREEN, fontsize=11, ha="center", va="top", path_effects=halo
    )
    d5_zs_ax.text(
        5.32, -1, "$Z_L$", color=GREEN, fontsize=11, ha="left", va="center", path_effects=halo
    )
    for ax in d5_grids:
        grid(ax, pad=0.78, d=5)
    for ax, title in zip(
        d5_grids,
        (
            "(j) 20 X gauges: horizontal pairs",
            "(k) 12 Z gauges: 8 blocks + 4 edge pairs",
            "(l) 12 X stabilizers: 8 blocks + 4 edge pairs",
            "(m) 4 Z stabilizers: two-row strips;\nlogical operators in green",
        ),
    ):
        ax.set_title(title, fontsize=10)

    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor=GOLD,
                markeredgecolor=INK,
                markersize=9,
                label="data qubit (Q label within patch)",
            ),
            Line2D(
                [],
                [],
                marker="D",
                color="w",
                markerfacecolor=X_RED,
                markersize=8,
                label="flag qubit, measures an X gauge",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor=Z_BLUE,
                markersize=9,
                label="syndrome qubit, measures a Z gauge",
            ),
            Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor="white",
                markeredgecolor=GREY,
                markeredgewidth=1.6,
                markersize=8,
                label="boundary relay qubit",
            ),
            Line2D([], [], color=X_RED, lw=2.8, label="X-gauge CX chain"),
            Line2D([], [], color=Z_BLUE, lw=2.8, label="Z-gauge CX chain"),
            Line2D(
                [],
                [],
                color="#8A8A8A",
                lw=1.9,
                ls="--",
                label="dashed = stabilizer (red X, blue Z)",
            ),
            Line2D([], [], color=GREEN, lw=3, label="$X_L$ (solid), $Z_L$ (dotted)"),
            Line2D(
                [],
                [],
                marker="o",
                color="w",
                markerfacecolor="#CFCFCF",
                markeredgecolor=GREY,
                markersize=7,
                label="unused fez qubit",
            ),
        ],
        loc="lower center",
        ncol=5,
        fontsize=9.5,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.035),
    )

    fig.suptitle(
        "distance-3 and distance-5 heavy-hex layouts on ibm_fez",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.966,
        "d=3: [[9,1,2,3]], 23 sites     |     d=5: [[25,1,8,5]], 65 sites     |     "
        "88 distinct physical qubits, all highlighted bonds verified",
        ha="center",
        fontsize=11,
        color=INK,
    )
    rule = ", ".join(f"q{F(qf)}\u2192Z on Q{k}" for qf, k in FLAG_RULE.items())
    fig.text(
        0.5,
        0.012,
        "d=3 reference: a Z round (4 syndromes + 8 flags) then an X round "
        f"(6 gauges) = 18 measurements. Deflagging: {rule}; the four relay "
        "flags are discarded.\n"
        "d=5 retains the same boundary relay arms: 57 core sites + 8 relays = 65. "
        "Connectivity is checked; gate order, timing, deflagging and the d=5 decoder "
        "still need validation.\n"
        "Operators: Sundaresan et al., Nat. Commun. 14, 2852 (2023), and "
        "Chamberland et al., PRX 10, 011022 (2020). Cached Fez map; no QPU execution.",
        ha="center",
        fontsize=8.2,
        color="#555555",
    )
    out = os.path.join(HERE, "heavyhex-blueprint.png")
    fig.savefig(out, dpi=160)
    plt.close(fig)
    manifest = {
        "distance": 5,
        "backend": "ibm_fez",
        "status": "connectivity-validated layout; not an executable circuit",
        "first_data_position": [3, 7],
        "data_qubits": d5.data,
        "x_gauges": d5.x_gauges,
        "z_gauges": d5.z_gauges,
        "x_stabilizers": d5.x_stabilizers,
        "z_stabilizers": d5.z_stabilizers,
        "x_gauge_ancillas": d5.x_ancillas,
        "z_gauge_ancillas": d5.z_ancillas,
        "boundary_relays": d5.relays,
        "couplings": d5.wires,
    }
    with open(os.path.join(HERE, "d5_fez_layout.json"), "w") as file:
        json.dump(manifest, file, indent=2)
        file.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
