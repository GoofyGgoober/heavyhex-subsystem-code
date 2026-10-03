"""Draw the paper's Figure 1: both patches on ibm_fez, and the placement test.

The patches go where heavyhex.patches.placement puts them for the given
calibration, so redraw on run day after `heavyhex calibrate`. Runs offline.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from draw_blueprint import (
    CALIBRATION,
    FEZ_MAP,
    GOLD,
    GREEN,
    GREY,
    HALO,
    INK,
    X_RED,
    Z_BLUE,
    DeviceMap,
    draw_chip_overview,
    draw_d3_device,
    draw_d5_device,
    load_device_map,
    patch_roles,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path as Curve
from matplotlib.transforms import Bbox

from heavyhex.patches.layout import HeavyHexLayout, build_layout
from heavyhex.patches.placement import (
    WEAK_CZ,
    Calibration,
    best_spots,
    broken_coupler,
    inside_qubits,
)

WEAK_BAND = "#C8C8C8"
TEST_DASH = (0, (5, 2.5))
GAP = 3  # points between a label and anything else
LABEL = dict(fontsize=9.5, fontweight="bold", color=INK, zorder=9, path_effects=HALO)


def taken_boxes(ax: Axes) -> list[Bbox]:
    """Screen boxes of the panel's labels and qubit markers."""
    boxes = [text.get_window_extent() for text in ax.texts]
    points = ax.figure.dpi / 72
    for markers in ax.collections:
        centers = ax.transData.transform(markers.get_offsets())
        for (x, y), size in zip(centers, np.broadcast_to(markers.get_sizes(), len(centers))):
            r = size**0.5 / 2 * points
            boxes.append(Bbox.from_extents(x - r, y - r, x + r, y + r))
    return boxes


def place(ax: Axes, text: str, candidates, taken: list[Bbox], **style) -> None:
    """Put text at the first (xy, dx, dy, ha, va) that stays in the panel and clears taken.

    Offsets are in points. If no candidate fits, the first one.
    """
    for i, (xy, dx, dy, ha, va) in enumerate([*candidates, candidates[0]]):
        label = ax.annotate(
            text, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, **style
        )
        box = label.get_window_extent()
        inside = all(ax.bbox.contains(x, y) for x, y in box.corners())
        clear = not any(box.padded(GAP * ax.figure.dpi / 72).overlaps(t) for t in taken)
        if i == len(candidates) or (inside and clear):
            taken.append(box)
            return
        label.remove()


def mark_weak_couplers(
    ax: Axes, device: DeviceMap, layout: HeavyHexLayout, calibration: Calibration, taken
) -> None:
    """Shade each coupler the patch uses whose CZ error is above WEAK_CZ, and print it."""
    for a, b in sorted({tuple(sorted((u, v))) for _, u, v in layout.couplings}):
        error = calibration.cz.get((a, b))
        if error is not None and error <= WEAK_CZ:
            continue
        (xa, ya), (xb, yb) = device.pos[a], device.pos[b]
        ax.plot([xa, xb], [ya, yb], color=WEAK_BAND, lw=10, solid_capstyle="butt", zorder=2)
        mid = ((xa + xb) / 2, (ya + yb) / 2)
        if ya == yb:
            candidates = [
                (mid, dx, dy, "center", va)
                for dy, va in ((-20, "top"), (20, "bottom"), (-34, "top"), (34, "bottom"))
                for dx in (0, -16, 16, -28, 28)
            ]
        else:
            candidates = [
                (mid, dx, dy, ha, "center")
                for dx, ha in ((9, "left"), (-9, "right"))
                for dy in (0, -8, 8)
            ]
        text = "broken" if broken_coupler(error) else f"{error:.1%}"
        place(ax, text, candidates, taken, **LABEL)


def rounded(corners: list[tuple[float, float]], radius: float) -> Curve:
    """Closed path through axis-aligned corners, each rounded off."""
    verts, codes = [], []
    for i, (x, y) in enumerate(corners):
        (px, py), (nx, ny) = corners[i - 1], corners[(i + 1) % len(corners)]
        before = (x - radius * np.sign(x - px), y - radius * np.sign(y - py))
        after = (x + radius * np.sign(nx - x), y + radius * np.sign(ny - y))
        verts += [before, (x, y), after]
        codes += [Curve.LINETO, Curve.CURVE3, Curve.CURVE3]
    codes[0] = Curve.MOVETO
    return Curve([*verts, verts[0]], [*codes, Curve.CLOSEPOLY])


def footprint(device: DeviceMap, qubits: list[int], side: float, above: float, below: float):
    """Corners of an outline hugging each row of the patch; its first 9 qubits are data."""
    rows = sorted({device.pos[q][1] for q in qubits[:9]})
    spans = [[device.pos[q][0] for q in qubits if device.pos[q][1] == y] for y in rows]
    cuts = [rows[0] - above, *((a + b) / 2 for a, b in zip(rows, rows[1:])), rows[-1] + below]
    right = [(max(s) + side, y) for i, s in enumerate(spans) for y in cuts[i : i + 2]]
    left = [(min(s) - side, y) for i, s in enumerate(spans) for y in cuts[i : i + 2]]
    ring = right + left[::-1]
    path = [p for i, p in enumerate(ring) if p != ring[i - 1]]
    return [
        p
        for i, p in enumerate(path)
        if not any(path[i - 1][k] == p[k] == path[(i + 1) % len(path)][k] for k in (0, 1))
    ]


def outline_placement_test(ax: Axes, device: DeviceMap, qubits: list[int], taken) -> list:
    """(c) Dashed outline around the placement test's qubits; its edges join taken."""
    corners = footprint(device, qubits, side=0.42, above=0.62, below=0.74)
    ax.add_patch(
        PathPatch(
            rounded(corners, 0.3),
            facecolor="none",
            edgecolor=GREEN,
            lw=1.8,
            ls=TEST_DASH,
            zorder=5,
        )
    )
    ends = ax.transData.transform(corners)
    stroke = 2 * ax.figure.dpi / 72
    taken += [Bbox(np.sort([a, b], 0)).padded(stroke) for a, b in zip(ends, np.roll(ends, -1, 0))]
    return corners


def label_placement_test(ax: Axes, corners: list, taken) -> None:
    """(c) Name the outline at whichever of its corners has room."""
    top, bottom = min(y for _, y in corners), max(y for _, y in corners)

    def end(y: float, pick) -> tuple[float, float]:
        return pick(x for x, row in corners if row == y), y

    place(
        ax,
        "placement test",
        [
            (end(bottom, max), 0, -6, "right", "top"),
            (end(bottom, min), 0, -6, "left", "top"),
            (end(top, min), 0, 6, "left", "bottom"),
            (end(top, max), 0, 6, "right", "bottom"),
        ],
        taken,
        **{**LABEL, "color": GREEN},
    )


def lift_z_labels(ax: Axes, layout: HeavyHexLayout) -> list[tuple[int, dict]]:
    """(c) Take each Z ancilla's q label off its vertical coupler, to go beside it as in (b)."""
    names = {f"q{q}": q for q in layout.z_ancillas.values()}
    lifted = []
    for text in [t for t in ax.texts if t.get_text() in names]:
        style = dict(fontsize=text.get_fontsize(), color=text.get_color())
        lifted.append((names[text.get_text()], style))
        text.remove()
    return lifted


def draw_key(fig: Figure, below: Axes) -> None:
    def marker(shape: str, face: str, edge: str, size: float, label: str, lw: float = 1.0):
        return Line2D(
            [],
            [],
            ls="none",
            marker=shape,
            markerfacecolor=face,
            markeredgecolor=edge,
            markeredgewidth=lw,
            markersize=size,
            label=label,
        )

    box = below.get_position()
    fig.legend(
        handles=[
            marker("o", GOLD, INK, 11, "data qubit"),
            marker("D", X_RED, "white", 9, "X ancilla, also a flag"),
            marker("o", Z_BLUE, "white", 10, "Z ancilla"),
            marker("o", "white", GREY, 10, "relay", lw=1.8),
            Line2D([], [], color=X_RED, lw=2.8, label="X-gauge CX"),
            Line2D([], [], color=Z_BLUE, lw=2.8, label="Z-gauge CX"),
            Line2D(
                [], [], color=WEAK_BAND, lw=10, label=f"coupler with CZ error above {WEAK_CZ:.0%}"
            ),
            Line2D([], [], color=GREEN, lw=1.8, ls=TEST_DASH, label="placement test (d = 3)"),
        ],
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=11,
        handlelength=2.6,
        columnspacing=2.2,
        bbox_to_anchor=((box.x0 + box.x1) / 2, box.y0 - 0.01),
    )


def build_figure(device: DeviceMap, calibration: Calibration) -> Figure:
    spots = best_spots(device.coords, device.edges, calibration)
    d3, d5 = (
        build_layout(
            d, device.coords, device.edges, origin=spots[d].origin, direction=spots[d].direction
        )
        for d in (3, 5)
    )
    d3_roles, d5_roles = patch_roles(d3), patch_roles(d5)
    test = inside_qubits(calibration)

    fig = plt.figure(figsize=(16, 9.4))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.36, 0.64],
        height_ratios=[1.2, 1],
        left=0.02,
        right=0.99,
        top=0.955,
        bottom=0.01,
        wspace=0.04,
        hspace=0.1,
    )
    ax_chip, ax_d3 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0])
    ax_d5 = fig.add_subplot(gs[:, 1])
    for ax in (ax_chip, ax_d3, ax_d5):
        ax.set_aspect("equal")
        ax.set_anchor("N")
        ax.axis("off")

    draw_chip_overview(ax_chip, device, ((d3, d3_roles), (d5, d5_roles)))
    draw_d3_device(ax_d3, device, d3, d3_roles)
    draw_d5_device(ax_d5, device, d5, d5_roles)
    for text in ax_d3.texts:
        if text.get_text().isdigit():
            text.set_text(f"q{text.get_text()}")
    ax_chip.set_title(f"(a) both patches on the {len(device.pos)}-qubit chip", fontsize=12)
    ax_d3.set_title(
        f"(b) d = 3: {len(d3.data)} data qubits, {len(d3.physical_qubits)} in all", fontsize=12
    )
    ax_d5.set_title(
        f"(c) d = 5: {len(d5.data)} data qubits, {len(d5.physical_qubits)} in all, "
        f"and the d = 3 placement test on {len(test)} of them",
        fontsize=12,
    )

    fig.draw_without_rendering()
    draw_key(fig, below=ax_d5)
    taken = taken_boxes(ax_d3)
    mark_weak_couplers(ax_d3, device, d3, calibration, taken)
    lifted = lift_z_labels(ax_d5, d5)
    taken = taken_boxes(ax_d5)
    corners = outline_placement_test(ax_d5, device, test, taken)
    for q, style in lifted:
        xy = device.pos[q]
        sides = [(xy, 9, 0, "left", "center"), (xy, -9, 0, "right", "center")]
        place(ax_d5, f"q{q}", sides, taken, path_effects=HALO, zorder=7, **style)
    mark_weak_couplers(ax_d5, device, d5, calibration, taken)
    label_placement_test(ax_d5, corners, taken)
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Draw the paper's Figure 1.")
    parser.add_argument(
        "--calibration",
        type=Path,
        default=CALIBRATION,
        metavar="PATH",
        help="calibration to place the patches from (default: the last `heavyhex calibrate`)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("figure1.png"),
        metavar="PATH",
        help="image to write; the suffix picks the format (default: ./figure1.png)",
    )
    args = parser.parse_args(argv)
    fig = build_figure(load_device_map(FEZ_MAP), Calibration.load(args.calibration))
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
