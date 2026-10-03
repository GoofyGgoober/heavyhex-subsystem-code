"""Draw the paper's Figure 1: the day's reps on ibm_fez.

The patches go where heavyhex.patches.placement's day plan puts them for the
given calibration, so redraw on run day after `heavyhex calibrate`. Rep 1 is
drawn in full, the other reps are boxed on the chip. Runs offline.
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
    GREY,
    HALO,
    INK,
    X_RED,
    Z_BLUE,
    DeviceMap,
    bounds,
    draw_chip_overview,
    draw_d3_device,
    draw_d5_device,
    load_device_map,
    patch_roles,
)
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Bbox

from heavyhex.patches.layout import HeavyHexLayout, build_layout
from heavyhex.patches.placement import (
    WEAK_CZ,
    Calibration,
    broken_coupler,
    day_plan,
)

WEAK_BAND = "#C8C8C8"
OTHER_REP = "#1B7A3D"  # boxes of the reps after the first
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
            Line2D([], [], color=OTHER_REP, lw=1.8, ls=":", label="a later rep's patch, in (a)"),
        ],
        loc="upper center",
        ncol=4,
        frameon=False,
        fontsize=11,
        handlelength=2.6,
        columnspacing=2.2,
        bbox_to_anchor=((box.x0 + box.x1) / 2, box.y0 - 0.01),
    )


def box_later_reps(ax: Axes, device: DeviceMap, plan: list) -> None:
    """(a) Box each patch of the reps after the first, labelled with its rep."""
    for r, jobs in enumerate(plan[1:], start=2):
        for job in jobs:
            for d, spot in job.items():
                layout = build_layout(
                    d, device.coords, device.edges, origin=spot.origin, direction=spot.direction
                )
                x0, x1, y0, y1 = bounds(device.pos, layout.physical_qubits)
                pad = 0.45  # inside rep 1's boxes, so the two stay apart
                ax.add_patch(
                    Rectangle(
                        (x0 - pad, y0 - pad),
                        x1 - x0 + 2 * pad,
                        y1 - y0 + 2 * pad,
                        facecolor="none",
                        edgecolor=OTHER_REP,
                        lw=1.8,
                        linestyle=":",
                        zorder=6,
                    )
                )
                alone = "" if len(jobs) == 1 else ", own job"
                ax.text(
                    x0 - pad,
                    y1 + pad + 0.25,
                    f"rep {r}: d={d}{alone}",
                    fontsize=8.5,
                    color=OTHER_REP,
                    fontweight="bold",
                    va="top",
                    path_effects=HALO,
                    zorder=9,
                )


def build_figure(device: DeviceMap, calibration: Calibration) -> Figure:
    plan = day_plan(device.coords, device.edges, calibration)
    if not plan:
        raise ValueError("no clean place for the d=5 patch: no rep to draw")
    (first, *_) = plan[0]
    spots = first if len(plan[0]) == 1 else {**plan[0][1], **plan[0][0]}
    d3, d5 = (
        build_layout(
            d, device.coords, device.edges, origin=spots[d].origin, direction=spots[d].direction
        )
        for d in (3, 5)
    )
    d3_roles, d5_roles = patch_roles(d3), patch_roles(d5)

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
    box_later_reps(ax_chip, device, plan)
    draw_d3_device(ax_d3, device, d3, d3_roles)
    draw_d5_device(ax_d5, device, d5, d5_roles)
    for text in ax_d3.texts:
        if text.get_text().isdigit():
            text.set_text(f"q{text.get_text()}")
    shown = "rep 1 drawn, the rest boxed" if len(plan) > 1 else "one rep"
    ax_chip.set_title(f"(a) the day's reps on the chip: {shown}", fontsize=12)
    ax_d3.set_title(
        f"(b) rep 1, d = 3: {len(d3.data)} data qubits, {len(d3.physical_qubits)} in all",
        fontsize=12,
    )
    ax_d5.set_title(
        f"(c) rep 1, d = 5: {len(d5.data)} data qubits, {len(d5.physical_qubits)} in all",
        fontsize=12,
    )

    fig.draw_without_rendering()
    draw_key(fig, below=ax_d5)
    taken = taken_boxes(ax_d3)
    mark_weak_couplers(ax_d3, device, d3, calibration, taken)
    lifted = lift_z_labels(ax_d5, d5)
    taken = taken_boxes(ax_d5)
    for q, style in lifted:
        xy = device.pos[q]
        sides = [(xy, 9, 0, "left", "center"), (xy, -9, 0, "right", "center")]
        place(ax_d5, f"q{q}", sides, taken, path_effects=HALO, zorder=7, **style)
    mark_weak_couplers(ax_d5, device, d5, calibration, taken)
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
