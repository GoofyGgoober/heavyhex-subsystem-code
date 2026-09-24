"""Draw the d=3 and d=5 patches on ibm_fez, and save the d=5 placement.

Writes heavyhex-blueprint.png and d5_fez_layout.json. Uses the saved coupling
map in fez_map.json, so it runs offline.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import NamedTuple

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle

from heavyhex.patches.layout import HeavyHexLayout, build_layout
from heavyhex.patches.operators import D3 as D3_CODE

HERE = Path(__file__).resolve().parent
FEZ_MAP = HERE / "fez_map.json"
PNG_NAME = "heavyhex-blueprint.png"
MANIFEST_NAME = "d5_fez_layout.json"

# Top-left data site of each patch on the cached map. (5, 1) reproduces the
# Falcon-27 placement of Fig. 4a, as tests/test_layout.py pins.
D3_ORIGIN = (5, 1)
D5_ORIGIN = (3, 7)

# The paper's d=3 flag rule, quoted in the caption: a lone flag on this X
# ancilla means Z on this data qubit (0-based id).
PAPER_FLAG_RULE = {"X2X5": 1, "X3X6": 5, "X4X7": 3, "X5X8": 7}

GOLD = "#F2C230"
GREEN = "#1B7A3D"
X_RED = "#E4574C"
Z_BLUE = "#2A78D6"
GREY = "#9A9A9A"
INK = "#333333"
IDLE = "#D8D8D8"
CHIP_BOND = "#E1E1E1"
PATCH_BOND = "#E2E2E2"
TINT = {Z_BLUE: "#CDDFF5", X_RED: "#F7D4D1"}
ROLE_COLOR = {"data": GOLD, "xanc": X_RED, "zanc": Z_BLUE, "relay": "white"}

HALO = [pe.withStroke(linewidth=2.6, foreground="white")]
WHITE_BOX = dict(facecolor="white", edgecolor="none", pad=0.12, alpha=0.85)
TINT_BOX = dict(facecolor=TINT[Z_BLUE], edgecolor="none", pad=0.12)
DASH = (5, 2.5)

# Label anchor per d=3 X stabilizer; the iteration order also sets the dash phase.
D3_X_STAB_LABEL = {
    "X1X2X4X5": (0.9, -0.42, "left", "bottom"),
    "X4X7": (3.1, -0.42, "right", "bottom"),
    "X3X6": (0.9, -3.58, "left", "top"),
    "X5X6X8X9": (3.1, -3.58, "right", "top"),
}


class DeviceMap(NamedTuple):
    coords: list[list[int]]
    edges: list[list[int]]
    pos: dict[int, tuple[float, float]]


def load_device_map(path: Path) -> DeviceMap:
    cached = json.loads(path.read_text())
    coords, edges = cached["coords"], cached["edges"]
    pos = {q: (float(x), float(y)) for q, (x, y) in enumerate(coords)}
    return DeviceMap(coords, edges, pos)


def mathtext(name: str) -> str:
    """X1X4 -> $X_{1}X_{4}$."""
    return "$" + re.sub(r"(\d+)", r"_{\1}", name) + "$"


def cell(label: int, distance: int = 3) -> tuple[int, int]:
    """Plot position of a 1-based code-qubit label: columns right, rows down."""
    return (label - 1) // distance + 1, -((label - 1) % distance + 1)


def centroid(support: tuple[int, ...], distance: int = 3) -> tuple[float, float]:
    points = [cell(label, distance) for label in support]
    return (
        sum(x for x, _ in points) / len(points),
        sum(y for _, y in points) / len(points),
    )


def patch_roles(layout: HeavyHexLayout) -> dict[int, tuple[str, str]]:
    """Physical qubit -> (role, operator or data name), in drawing order."""
    roles = {q: ("data", f"Q{k}") for k, q in layout.data.items()}
    roles.update((q, ("xanc", name)) for name, q in layout.x_ancillas.items())
    roles.update((q, ("zanc", name)) for name, q in layout.z_ancillas.items())
    roles.update((q, ("relay", "")) for q in layout.relays)
    return roles


def bounds(pos: dict[int, tuple[float, float]], qubits) -> tuple[float, float, float, float]:
    """(xmin, xmax, ymin, ymax) of the given physical qubits."""
    xs = [pos[q][0] for q in qubits]
    ys = [pos[q][1] for q in qubits]
    return min(xs), max(xs), min(ys), max(ys)


def bond_color(kind: str) -> str:
    return Z_BLUE if kind == "z" else X_RED


def code_grid(ax: Axes, distance: int = 3, pad: float = 0.62) -> None:
    """The d*d numbered code qubits, on top of whatever operators were drawn."""
    for label in range(1, distance * distance + 1):
        x, y = cell(label, distance)
        ax.scatter([x], [y], s=165, color=GOLD, edgecolor=INK, lw=1.1, zorder=6)
        ax.text(x, y - 0.015, f"{label}", ha="center", va="center", fontsize=8.5, zorder=7)
    ax.set_xlim(1 - pad, distance + pad)
    ax.set_ylim(-distance - pad, -1 + pad)


def operator_box(
    ax: Axes,
    support: tuple[int, ...],
    color: str,
    *,
    fill: bool,
    pad: float,
    lw: float = 1.8,
    ls: str | tuple = "-",
    z: int = 3,
    distance: int = 3,
) -> None:
    """Rounded outline around a block-shaped operator support."""
    points = [cell(label, distance) for label in support]
    x0 = min(x for x, _ in points) - pad
    y0 = min(y for _, y in points) - pad
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0),
            max(x for x, _ in points) - x0 + pad,
            max(y for _, y in points) - y0 + pad,
            boxstyle="round,pad=0.0,rounding_size=0.16",
            facecolor=TINT[color] if fill else "none",
            edgecolor=color,
            linewidth=lw,
            linestyle=ls,
            zorder=z,
        )
    )


def operator_bar(
    ax: Axes,
    support: tuple[int, ...],
    color: str,
    label: str,
    *,
    distance: int = 3,
    z: int = 3,
    offset: tuple[float, float] = (0.17, 0.0),
    ha: str = "left",
    va: str = "center",
) -> None:
    """Weight-2 operator: a thick link with its ancilla marker at the midpoint."""
    (x1, y1), (x2, y2) = cell(support[0], distance), cell(support[1], distance)
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
        mx + offset[0],
        my + offset[1],
        label,
        ha=ha,
        va=va,
        fontsize=8.2,
        color=color,
        fontweight="bold",
        zorder=z + 4,
        bbox=WHITE_BOX,
    )


def draw_chip_overview(
    ax: Axes, device: DeviceMap, patches: tuple[tuple[HeavyHexLayout, dict], ...]
) -> None:
    """(a) Whole chip in grey with both patches highlighted and boxed."""
    for a, b in device.edges:
        ax.plot(*zip(device.pos[a], device.pos[b]), color=CHIP_BOND, lw=0.6, zorder=1)
    ax.scatter(*zip(*device.pos.values()), s=5, color=IDLE, zorder=2)
    for layout, roles in patches:
        for kind, u, v in layout.couplings:
            ax.plot(*zip(device.pos[u], device.pos[v]), color=bond_color(kind), lw=1.5, zorder=3)
        for q, (role, _) in roles.items():
            color = ROLE_COLOR[role]
            ax.scatter(*device.pos[q], s=20, color=color, edgecolor=INK, lw=0.5, zorder=4)
        x0, x1, y0, y1 = bounds(device.pos, roles)
        ax.add_patch(
            Rectangle(
                (x0 - 0.7, y0 - 0.7),
                x1 - x0 + 1.4,
                y1 - y0 + 1.4,
                facecolor="none",
                edgecolor=INK,
                lw=1.2,
                linestyle="--",
                zorder=5,
            )
        )
    ax.set_xlim(0.4, 16.6)
    ax.set_ylim(16.0, -0.4)  # Include the lower patch boundary below device row 15.
    for layout, roles in patches:
        x0, x1, y0, y1 = bounds(device.pos, roles)
        ax.text(
            x1 + 1.0,
            (y0 + y1) / 2,
            f"d={layout.distance}\n{len(layout.physical_qubits)} sites",
            fontsize=9,
            va="center",
            fontweight="bold",
            bbox=WHITE_BOX,
        )
    used = sum(len(roles) for _, roles in patches)
    ax.set_title(
        f"(a) both patches: {used} of {len(device.pos)} sites; no shared qubits", fontsize=10
    )


def draw_d3_device(ax: Axes, device: DeviceMap, d3: HeavyHexLayout, roles: dict) -> None:
    """(b) The d=3 patch where it sits on the chip, with its idle neighbours."""
    x0, x1, y0, y1 = bounds(device.pos, roles)
    left, right, bottom, top = x0 - 1.3, x1 + 1.3, y0 - 1.0, y1 + 1.35
    bonds = {tuple(sorted(edge)) for edge in device.edges}
    idle = sorted(
        w
        for q in roles
        for w in device.pos
        if w not in roles
        and left <= device.pos[w][0] <= right
        and bottom <= device.pos[w][1] <= top
        and tuple(sorted((q, w))) in bonds
    )
    shown = set(roles) | set(idle)
    for a, b in device.edges:
        if {a, b} <= shown:
            ax.plot(*zip(device.pos[a], device.pos[b]), color=PATCH_BOND, lw=1.4, zorder=1)
    ax.scatter(
        [device.pos[q][0] for q in idle],
        [device.pos[q][1] for q in idle],
        s=30,
        color=IDLE,
        zorder=2,
    )
    for kind, u, v in d3.couplings:
        ax.plot(
            *zip(device.pos[u], device.pos[v]),
            color=bond_color(kind),
            lw=2.8,
            solid_capstyle="round",
            zorder=3,
        )
    for q, (role, name) in roles.items():
        x, y = device.pos[q]
        if role == "data":
            ax.scatter([x], [y], s=170, color=GOLD, edgecolor=INK, lw=1.1, zorder=6)
            ax.text(
                x,
                y - 0.22,
                f"$Q_{{{name[1:]}}}$",
                ha="center",
                va="bottom",
                fontsize=12,
                fontweight="bold",
                zorder=7,
                path_effects=HALO,
            )
        elif role == "relay":
            ax.scatter([x], [y], s=120, color="white", edgecolor=GREY, lw=1.8, zorder=6)
        else:
            color = ROLE_COLOR[role]
            ax.scatter(
                [x],
                [y],
                s=130,
                color=color,
                edgecolor="white",
                lw=1.0,
                marker="D" if role == "xanc" else "o",
                zorder=6,
            )
            dx, dy, ha, va = (
                (0.28, 0.28, "left", "top") if role == "xanc" else (-0.26, 0.0, "right", "center")
            )
            ax.text(
                x + dx,
                y + dy,
                mathtext(name),
                ha=ha,
                va=va,
                fontsize=10,
                color=color,
                fontweight="bold",
                zorder=8,
                bbox=WHITE_BOX,
            )
        dx, dy, ha, va = {
            "data": (0.0, 0.17, "center", "top"),
            "xanc": (0.18, -0.17, "left", "bottom"),
            "zanc": (0.26, 0.0, "left", "center"),
            "relay": (0.15, -0.15, "left", "bottom"),
        }[role]
        ax.text(
            x + dx,
            y + dy,
            f"{q}",
            ha=ha,
            va=va,
            fontsize=7.5,
            color=GREY,
            zorder=7,
            path_effects=HALO,
        )
    ax.set_xlim(left, right)
    ax.set_ylim(top, bottom)
    ax.set_title("(b) d=3 coupling layout — every link is a real fez bond", fontsize=10)


def draw_d3_operator_table(ax: Axes, d3: HeavyHexLayout) -> None:
    """(c) Measured gauges with their ancillas, then the inferred stabilizers."""
    rows = [("head", f"{len(d3.z_gauges)} Z gauges measured (syndrome on a vertical link)", "")]
    rows += [
        ("z", mathtext(name), f"q{d3.z_ancillas[name]}")
        for name in ("Z1Z2", "Z2Z3Z5Z6", "Z4Z5Z7Z8", "Z8Z9")
    ]
    rows += [("head", f"{len(d3.x_gauges)} X gauges measured (in-row flag qubit)", "")]
    rows += [("x", mathtext(name), f"q{d3.x_ancillas[name]}") for name in d3.x_gauges]
    rows += [
        ("head", f"{len(d3.z_stabilizers)} Z stabilizers, inferred as gauge products", ""),
        ("z", mathtext("Z1Z2Z4Z5Z7Z8"), f"= {mathtext('Z1Z2')}·{mathtext('Z4Z5Z7Z8')}"),
        ("z", mathtext("Z2Z3Z5Z6Z8Z9"), f"= {mathtext('Z2Z3Z5Z6')}·{mathtext('Z8Z9')}"),
        ("head", f"{len(d3.x_stabilizers)} X stabilizers, inferred as gauge products", ""),
        ("x", mathtext("X1X2X4X5"), f"= {mathtext('X1X4')}·{mathtext('X2X5')}"),
        ("x", mathtext("X5X6X8X9"), f"= {mathtext('X5X8')}·{mathtext('X6X9')}"),
        ("x", mathtext("X4X7"), "= gauge, already central"),
        ("x", mathtext("X3X6"), "= gauge, already central"),
        ("head", "logical operators", ""),
        ("g", f"$X_L$ = {mathtext('X1X2X3')}", "column 1 of the grid"),
        ("g", f"$Z_L$ = {mathtext('Z1Z4Z7')}", "row 1 of the grid"),
    ]
    color = {"z": Z_BLUE, "x": X_RED, "g": GREEN}
    y = 1.0
    for kind, left, right in rows:
        if kind == "head":
            y -= 0.012
            ax.text(
                0.0,
                y,
                left,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.2,
                color=INK,
                fontweight="bold",
            )
            y -= 0.054
            continue
        for x, text in ((0.035, left), (0.40, right)):
            ax.text(
                x,
                y,
                text,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9.8,
                color=color[kind],
            )
        y -= 0.0455
    ax.set_title("(c) d=3 operators and ancilla assignments", fontsize=10)


def draw_d3_x_gauges(ax: Axes, d3: HeavyHexLayout) -> None:
    """(d) The X gauges: pairs along each row, each with its own ancilla."""
    for name, support in d3.x_gauges.items():
        operator_bar(
            ax,
            support,
            X_RED,
            f"q{d3.x_ancillas[name]}",
            offset=(0.0, 0.21),
            ha="center",
            va="bottom",
        )
    code_grid(ax)
    ax.set_title(f"(d) {len(d3.x_gauges)} X gauges measured, one flag qubit each", fontsize=10)


def draw_d3_z_gauges(ax: Axes, d3: HeavyHexLayout) -> None:
    """(e) Two boundary column pairs and two 2x2 blocks."""
    for name, support in d3.z_gauges.items():
        if len(support) == 2:
            operator_bar(ax, support, Z_BLUE, f"q{d3.z_ancillas[name]}", z=4)
            continue
        # Only one block is tinted; the other stays an outline so they read apart.
        filled = name == "Z4Z5Z7Z8"
        operator_box(
            ax,
            support,
            Z_BLUE,
            fill=filled,
            pad=0.30 if filled else 0.21,
            lw=1.8 if filled else 2.4,
            z=2 if filled else 5,
        )
        x, y = centroid(support)
        ax.scatter([x], [y], s=115, color=Z_BLUE, edgecolor="white", lw=1.0, zorder=7)
        ax.text(
            x + 0.17,
            y,
            f"q{d3.z_ancillas[name]}",
            ha="left",
            va="center",
            fontsize=8.2,
            color=Z_BLUE,
            fontweight="bold",
            zorder=8,
            bbox=TINT_BOX if filled else WHITE_BOX,
        )
    code_grid(ax)
    ax.set_title(f"(e) {len(d3.z_gauges)} Z gauges measured, one syndrome qubit each", fontsize=10)


def draw_d3_x_stabilizers(ax: Axes, d3: HeavyHexLayout) -> None:
    """(f) Two gauge products plus the two gauges that are already central."""
    for i, (name, (lx, ly, ha, va)) in enumerate(D3_X_STAB_LABEL.items()):
        support = d3.x_stabilizers[name]
        operator_box(
            ax,
            support,
            X_RED,
            fill=False,
            pad=0.32 if len(support) == 4 else 0.16,
            lw=1.9,
            ls=(0 if i % 2 == 0 else 2.5, DASH),
            z=4,
        )
        ax.text(
            lx,
            ly,
            mathtext(name),
            color=X_RED,
            fontsize=8.6,
            fontweight="bold",
            ha=ha,
            va=va,
            zorder=7,
        )
    code_grid(ax, pad=0.75)
    ax.set_title(
        f"(f) {len(d3.x_stabilizers)} X stabilizers: two are gauge products, two are\n"
        "gauges that are already central",
        fontsize=10,
    )


def draw_d3_z_stabilizers(ax: Axes, d3: HeavyHexLayout) -> None:
    """(g) The two weight-6 strips, plus the logical operators."""
    for i, support in enumerate(d3.z_stabilizers.values()):
        operator_box(
            ax,
            support,
            Z_BLUE,
            fill=i == 0,
            pad=0.36 - 0.19 * i,
            lw=1.9,
            ls=(0 if i == 0 else 2.5, DASH),
            z=2 + i,
        )
    for y, name, bbox in ((-1.5, "Z1Z2Z4Z5Z7Z8", TINT_BOX), (-2.5, "Z2Z3Z5Z6Z8Z9", WHITE_BOX)):
        ax.text(
            2.35,
            y,
            mathtext(name),
            color=Z_BLUE,
            fontsize=8.6,
            fontweight="bold",
            ha="center",
            va="center",
            zorder=7,
            bbox=bbox,
        )
    for logical, ls in ((D3_CODE.logical_x, "-"), (D3_CODE.logical_z, ":")):
        ax.plot(
            [cell(label)[0] for label in logical],
            [cell(label)[1] for label in logical],
            color=GREEN,
            lw=3.4,
            ls=ls,
            solid_capstyle="round",
            zorder=5,
        )
    code_grid(ax, pad=0.75)
    for x, y, text, ha, va in (
        (1, -3.42, "$X_L$", "center", "top"),
        (3.3, -1, "$Z_L$", "left", "center"),
    ):
        ax.text(
            x,
            y,
            text,
            color=GREEN,
            fontsize=11,
            fontweight="bold",
            ha=ha,
            va=va,
            zorder=7,
            path_effects=HALO,
        )
    ax.set_title(
        f"(g) {len(d3.z_stabilizers)} Z stabilizers, both gauge products;\n"
        "logical operators in green",
        fontsize=10,
    )


def draw_d5_device(ax: Axes, device: DeviceMap, d5: HeavyHexLayout, roles: dict) -> None:
    """(h) The d=5 patch where it sits on the chip."""
    x0, x1, y0, y1 = bounds(device.pos, roles)
    left, right, bottom, top = x0 - 1.0, x1 + 1.0, y0 - 0.8, y1 + 0.8
    for a, b in device.edges:
        if all(
            left <= device.pos[q][0] <= right and bottom <= device.pos[q][1] <= top for q in (a, b)
        ):
            ax.plot(*zip(device.pos[a], device.pos[b]), color=PATCH_BOND, lw=1.2, zorder=1)
    for kind, u, v in d5.couplings:
        ax.plot(
            *zip(device.pos[u], device.pos[v]),
            color=bond_color(kind),
            lw=2.8,
            solid_capstyle="round",
            zorder=3,
        )
    for q, (role, name) in roles.items():
        x, y = device.pos[q]
        color = ROLE_COLOR[role]
        ax.scatter(
            x,
            y,
            s=135 if role == "data" else 95,
            color=color,
            edgecolor=INK if role == "data" else GREY if role == "relay" else "white",
            lw=1.1,
            marker="D" if role == "xanc" else "o",
            zorder=6,
        )
        if role == "data":
            ax.text(
                x,
                y - 0.23,
                f"$Q_{{{name[1:]}}}$",
                ha="center",
                va="bottom",
                fontsize=10.5,
                zorder=7,
                path_effects=HALO,
            )
        ax.text(
            x,
            y + 0.2,
            f"q{q}",
            ha="center",
            va="top",
            fontsize=7.4,
            color=GREY if role in ("data", "relay") else color,
            path_effects=HALO,
            zorder=7,
        )
    ax.set_xlim(left, right)
    ax.set_ylim(top, bottom)
    ax.set_title(
        f"(h) d=5 coupling layout: {len(d5.data)} data + {len(d5.x_ancillas)} X ancillas + "
        f"{len(d5.z_ancillas)} Z ancillas + {len(d5.relays)} relays "
        f"= {len(d5.physical_qubits)} sites",
        fontsize=10,
    )


def draw_d5_operator_table(ax: Axes, d5: HeavyHexLayout) -> None:
    """(i) Every d=5 gauge and its ancilla."""
    columns = (
        (0.0, f"{len(d5.x_gauges)} X gauges", d5.x_gauges, d5.x_ancillas, X_RED),
        (0.49, f"{len(d5.z_gauges)} Z gauges", d5.z_gauges, d5.z_ancillas, Z_BLUE),
    )
    for left, title, gauges, ancillas, color in columns:
        ax.text(
            left,
            1.0,
            title,
            transform=ax.transAxes,
            fontsize=10,
            fontweight="bold",
            color=INK,
            va="top",
        )
        for index, name in enumerate(gauges):
            y = 0.94 - index * 0.046
            ax.text(
                left, y, mathtext(name), transform=ax.transAxes, fontsize=9.6, color=color, va="top"
            )
            ax.text(
                left + 0.42,
                y,
                f"q{ancillas[name]}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9.6,
                color=color,
            )
    ax.text(
        0.49,
        0.32,
        f"Inferred: {len(d5.x_stabilizers)} X + {len(d5.z_stabilizers)} Z stabilizers\n"
        "One logical qubit; 8 gauge qubits\n"
        "Q labels restart within each patch.\n"
        "q labels are physical Fez IDs.",
        transform=ax.transAxes,
        va="top",
        fontsize=9.5,
        linespacing=1.6,
        color=INK,
    )
    ax.text(
        0.49,
        0.11,
        "$X_L$ = " + mathtext("X1X2X3X4X5") + "\n$Z_L$ = " + mathtext("Z1Z6Z11Z16Z21"),
        transform=ax.transAxes,
        va="top",
        fontsize=10,
        linespacing=1.7,
        color=GREEN,
    )
    ax.set_title("(i) d=5 operators and ancilla assignments", fontsize=10)


def draw_d5_grids(axes: tuple[Axes, ...], d5: HeavyHexLayout) -> None:
    """(j)-(m) The same four views as the d=3 row."""
    ax_xg, ax_zg, ax_xs, ax_zs = axes
    for name, support in d5.x_gauges.items():
        operator_bar(
            ax_xg,
            support,
            X_RED,
            f"q{d5.x_ancillas[name]}",
            distance=5,
            offset=(0.0, 0.23),
            ha="center",
            va="bottom",
        )
    for name, support in d5.z_gauges.items():
        if len(support) == 2:
            on_left = support[0] <= 5
            operator_bar(
                ax_zg,
                support,
                Z_BLUE,
                f"q{d5.z_ancillas[name]}",
                distance=5,
                offset=(-0.20, 0.0) if on_left else (0.20, 0.0),
                ha="right" if on_left else "left",
            )
            continue
        operator_box(ax_zg, support, Z_BLUE, fill=True, pad=0.15, distance=5)
        x, y = centroid(support, 5)
        ax_zg.text(
            x,
            y,
            f"q{d5.z_ancillas[name]}",
            color=Z_BLUE,
            fontsize=8.2,
            ha="center",
            va="center",
            bbox=TINT_BOX,
            zorder=7,
        )
    for support in d5.x_stabilizers.values():
        operator_box(
            ax_xs,
            support,
            X_RED,
            fill=False,
            pad=0.19 if len(support) == 4 else 0.13,
            ls="--",
            distance=5,
        )
    for index, support in enumerate(d5.z_stabilizers.values()):
        operator_box(
            ax_zs,
            support,
            Z_BLUE,
            fill=False,
            pad=0.30 - 0.10 * (index % 2),
            ls=(0 if index % 2 == 0 else 2.5, DASH),
            distance=5,
        )
        ax_zs.text(
            5.35,
            -index - 1.5,
            f"$S^Z_{{{index + 1}}}$",
            color=Z_BLUE,
            fontsize=9,
            ha="left",
            va="center",
            zorder=7,
        )
    ax_zs.plot([1, 1], [-1, -5], color=GREEN, lw=3.4, zorder=5)
    ax_zs.plot([1, 5], [-1, -1], color=GREEN, lw=3.4, ls=":", zorder=5)
    ax_zs.text(
        1, -5.42, "$X_L$", color=GREEN, fontsize=11, ha="center", va="top", path_effects=HALO
    )
    ax_zs.text(
        5.32, -1, "$Z_L$", color=GREEN, fontsize=11, ha="left", va="center", path_effects=HALO
    )
    for ax in axes:
        code_grid(ax, distance=5, pad=0.78)
    titles = (
        f"(j) {len(d5.x_gauges)} X gauges: horizontal pairs",
        f"(k) {len(d5.z_gauges)} Z gauges: 8 blocks + 4 edge pairs",
        f"(l) {len(d5.x_stabilizers)} X stabilizers: 8 blocks + 4 edge pairs",
        f"(m) {len(d5.z_stabilizers)} Z stabilizers: two-row strips;\nlogical operators in green",
    )
    for ax, title in zip(axes, titles):
        ax.set_title(title, fontsize=10)


def draw_legend(fig: Figure) -> None:
    def dot(facecolor: str, label: str, **style) -> Line2D:
        return Line2D(
            [], [], marker="o", color="w", markerfacecolor=facecolor, label=label, **style
        )

    fig.legend(
        handles=[
            dot(GOLD, "data qubit (Q label within patch)", markeredgecolor=INK, markersize=9),
            Line2D(
                [],
                [],
                marker="D",
                color="w",
                markerfacecolor=X_RED,
                markersize=8,
                label="flag qubit, measures an X gauge",
            ),
            dot(Z_BLUE, "syndrome qubit, measures a Z gauge", markersize=9),
            dot(
                "white",
                "boundary relay qubit",
                markeredgecolor=GREY,
                markeredgewidth=1.6,
                markersize=8,
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
            dot("#CFCFCF", "unused fez qubit", markeredgecolor=GREY, markersize=7),
        ],
        loc="lower center",
        ncol=5,
        fontsize=9.5,
        framealpha=0.95,
        bbox_to_anchor=(0.5, 0.035),
    )


def draw_captions(fig: Figure, d3: HeavyHexLayout, d5: HeavyHexLayout) -> None:
    fig.suptitle(
        "distance-3 and distance-5 heavy-hex layouts on ibm_fez",
        fontsize=16,
        fontweight="bold",
        y=0.985,
    )
    fig.text(
        0.5,
        0.966,
        f"d=3: [[9,1,2,3]], {len(d3.physical_qubits)} sites     |     "
        f"d=5: [[25,1,8,5]], {len(d5.physical_qubits)} sites     |     "
        f"{len(d3.physical_qubits | d5.physical_qubits)} distinct physical qubits, "
        "all highlighted bonds verified",
        ha="center",
        fontsize=11,
        color=INK,
    )
    rule = ", ".join(f"q{d3.x_ancillas[name]}→Z on Q{q + 1}" for name, q in PAPER_FLAG_RULE.items())
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


def build_figure(device: DeviceMap, d3: HeavyHexLayout, d5: HeavyHexLayout) -> Figure:
    d3_roles, d5_roles = patch_roles(d3), patch_roles(d5)
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
    d3_grids = tuple(fig.add_subplot(gs[1, i : i + 3]) for i in range(0, 12, 3))
    ax_d5 = fig.add_subplot(gs[2, 0:7])
    ax_d5_txt = fig.add_subplot(gs[2, 7:12])
    d5_grids = tuple(fig.add_subplot(gs[3, i : i + 3]) for i in range(0, 12, 3))

    grids = d3_grids + d5_grids
    for ax in (ax_chip, ax_dev, ax_txt, ax_d5, ax_d5_txt) + grids:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    for ax in (ax_chip, ax_dev, ax_d5) + grids:
        ax.set_aspect("equal")
        ax.set_anchor("N")

    draw_chip_overview(ax_chip, device, ((d3, d3_roles), (d5, d5_roles)))
    draw_d3_device(ax_dev, device, d3, d3_roles)
    draw_d3_operator_table(ax_txt, d3)
    draw_d3_x_gauges(d3_grids[0], d3)
    draw_d3_z_gauges(d3_grids[1], d3)
    draw_d3_x_stabilizers(d3_grids[2], d3)
    draw_d3_z_stabilizers(d3_grids[3], d3)
    draw_d5_device(ax_d5, device, d5, d5_roles)
    draw_d5_operator_table(ax_d5_txt, d5)
    draw_d5_grids(d5_grids, d5)
    draw_legend(fig)
    draw_captions(fig, d3, d5)
    return fig


def write_manifest(layout: HeavyHexLayout, path: Path) -> None:
    manifest = {
        "distance": layout.distance,
        "backend": "ibm_fez",
        "status": "connectivity-validated layout; circuit generated by heavyhex.circuits.flagged",
        "first_data_position": list(D5_ORIGIN),
        "data_qubits": layout.data,
        "x_gauges": layout.x_gauges,
        "z_gauges": layout.z_gauges,
        "x_stabilizers": layout.x_stabilizers,
        "z_stabilizers": layout.z_stabilizers,
        "x_gauge_ancillas": layout.x_ancillas,
        "z_gauge_ancillas": layout.z_ancillas,
        "boundary_relays": layout.relays,
        "couplings": layout.couplings,
    }
    path.write_text(json.dumps(manifest, indent=2) + "\n")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Draw the heavy-hex blueprint figure.")
    parser.add_argument(
        "--out",
        type=Path,
        default=HERE,
        metavar="DIR",
        help=f"directory for {PNG_NAME} and {MANIFEST_NAME} (default: next to this script)",
    )
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)

    device = load_device_map(FEZ_MAP)
    d3 = build_layout(3, device.coords, device.edges, origin=D3_ORIGIN)
    d5 = build_layout(5, device.coords, device.edges, origin=D5_ORIGIN)
    if not d3.physical_qubits.isdisjoint(d5.physical_qubits):
        raise ValueError("the two patches must not share any physical qubits")
    for layout in (d3, d5):
        print(
            f"d={layout.distance}: {len(layout.physical_qubits)} qubits, "
            f"{len(layout.couplings)} real ibm_fez bonds"
        )

    fig = build_figure(device, d3, d5)
    png = args.out / PNG_NAME
    fig.savefig(png, dpi=160)
    plt.close(fig)
    write_manifest(d5, args.out / MANIFEST_NAME)
    print(f"wrote {png}")


if __name__ == "__main__":
    main()
