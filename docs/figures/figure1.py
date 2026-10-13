"""Draw the day's reps on ibm_fez: the paper's chip figure, paper/figures/layout.pdf.

paper/figures/make_figures.py draws that from day 1's calibration. Run on its own,
this draws figure1.png from the last `heavyhex calibrate`, as on the run days. The
patches go where heavyhex.patches.placement's day plan puts them for the given
calibration. Rep 1 is drawn in full, the other reps are boxed on the chip. Runs offline.

    python docs/figures/figure1.py [--calibration PATH] [--out PATH] [--weak Q,Q,...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import Bbox

from heavyhex.patches.layout import HeavyHexLayout, build_layout
from heavyhex.patches.placement import (
    WEAK_CZ,
    Calibration,
    broken_coupler,
    day_plan,
)

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

WEAK_BAND = "#C8C8C8"
WEAK_RING = "#6F6F6F"
OTHER_REP = "#1B7A3D"  # boxes of the reps after the first
QUBIT_NUMBER = "#6B6B6B"  # the Fez qubit numbers, darker than the blueprint's grey for print
GAP = 3  # points between a label and anything else
# The paper prints the figure at its text width, PRINT_WIDTH, so every size here is scaled by
# PRINT_WIDTH / WIDTH; MIN_PT keeps the smallest label at 7 pt in print.
WIDTH, HEIGHT, PRINT_WIDTH = 12.0, 7.05, 7.0
MIN_PT = 7.0 * WIDTH / PRINT_WIDTH
LABEL = {"fontsize": MIN_PT, "fontweight": "bold", "color": INK, "zorder": 9, "path_effects": HALO}
LEADER = {"arrowstyle": "-", "lw": 0.8, "color": INK, "shrinkA": 0, "shrinkB": 3}
# A white box under each Fez qubit number, so that no coupler line runs through it.
UNDER = {"boxstyle": "square,pad=0.08", "facecolor": "white", "edgecolor": "none"}
TITLE = 13.5


def taken_boxes(ax: Axes) -> list[Bbox]:
    """Screen boxes of the panel's labels and qubit markers."""
    boxes = [text.get_window_extent() for text in ax.texts]
    points = ax.figure.dpi / 72
    for markers in ax.collections:
        centers = ax.transData.transform(markers.get_offsets())
        sizes = markers.get_sizes()  # pyright: ignore[reportAttributeAccessIssue]
        for (x, y), size in zip(centers, np.broadcast_to(sizes, len(centers))):
            r = size**0.5 / 2 * points
            boxes.append(Bbox.from_extents(x - r, y - r, x + r, y + r))
    return boxes


def place(ax: Axes, text: str, candidates, taken: list[Bbox], **style) -> None:
    """Put text at the first (xy, dx, dy, ha, va) that stays in the panel and clears taken.

    Offsets are in points. If no candidate fits, the one in the panel whose text overlaps the
    taken boxes least (beside a short coupler, its own end's qubit number can block every
    leader).
    """
    pad = GAP * ax.figure.dpi / 72

    def put(xy, dx, dy, ha, va):
        return ax.annotate(
            text, xy, xytext=(dx, dy), textcoords="offset points", ha=ha, va=va, **style
        )

    fallback, least = candidates[0], float("inf")
    for candidate in candidates:
        label = put(*candidate)
        box = label.get_window_extent()
        if all(ax.bbox.contains(x, y) for x, y in box.corners()):
            if not any(box.padded(pad).overlaps(t) for t in taken):
                taken.append(box)
                return
            words = Text.get_window_extent(label)  # the text alone, without its leader
            overlap = sum(
                max(0.0, min(words.x1 + pad, t.x1) - max(words.x0 - pad, t.x0))
                * max(0.0, min(words.y1 + pad, t.y1) - max(words.y0 - pad, t.y0))
                for t in taken
            )
            if overlap < least:
                fallback, least = candidate, overlap
        label.remove()
    taken.append(put(*fallback).get_window_extent())


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
                for dy in (0, -8, 8, -16, 16, -24, -32)
            ]
        text = "broken" if broken_coupler(error) else f"{error:.1%}"
        place(ax, text, candidates, taken, arrowprops=LEADER, **LABEL)


def mark_weak_qubits(ax: Axes, device: DeviceMap, layout: HeavyHexLayout, qubits, size) -> None:
    """Ring each of the given qubits that the patch uses (the caption says why it is weak)."""
    for q in sorted(set(qubits) & layout.physical_qubits):
        x, y = device.pos[q]
        ax.scatter(
            [x], [y], s=size * 3.2, facecolor="none", edgecolor=WEAK_RING, lw=1.6, zorder=5.8
        )


def flags(layout: HeavyHexLayout) -> set[int]:
    """The X ancillas that a bulk Z gauge borrows as flags (Z-gauge couplings to an X ancilla)."""
    x_ancillas = set(layout.x_ancillas.values())
    return {q for kind, a, b in layout.couplings if kind == "z" for q in (a, b) if q in x_ancillas}


def mark_flags(ax: Axes, device: DeviceMap, layout: HeavyHexLayout, size: float) -> None:
    """Redraw the X ancillas that also serve as flags with a dark edge, at the panel's size."""
    xy = np.array([device.pos[q] for q in sorted(flags(layout))])
    ax.scatter(
        xy[:, 0], xy[:, 1], s=size, color=X_RED, edgecolor=INK, lw=1.5, marker="D", zorder=6.5
    )


def keep_corner_data_labels(ax: Axes, distance: int) -> None:
    """(c) Keep the Q labels of the four corner data qubits only; the numbering runs down each
    column (Fig. 1 caption), and 25 labels crowd the panel."""
    corners = {1, distance, distance * (distance - 1) + 1, distance * distance}
    for text in list(ax.texts):
        content = text.get_text()
        if content.startswith("$Q_{") and int(content[4:-2]) not in corners:
            text.remove()


def tidy_d3_labels(ax: Axes, device: DeviceMap, roles: dict) -> None:
    """(b) Make room for print-size labels: drop the X-gauge names, which the drawing already
    shows (each red diamond joins two data qubits), put the data and X-ancilla numbers under
    their markers and the relay numbers on the outer side of theirs."""
    middle = sum(device.pos[q][0] for q in roles) / len(roles)
    for text in list(ax.texts):
        content = text.get_text()
        if content.startswith("$X"):
            text.remove()
        elif content.isdigit() and int(content) in roles:
            q = int(content)
            x, y = device.pos[q]
            role = roles[q][0]
            if role in ("data", "xanc"):
                text.set_position((x, y + 0.2))
                text.set_ha("center")
                text.set_va("top")
            elif role == "relay":
                side = 1 if x > middle else -1
                text.set_position((x + 0.24 * side, y))
                text.set_ha("left" if side > 0 else "right")
                text.set_va("center")


def lift_z_labels(ax: Axes, layout: HeavyHexLayout) -> list[tuple[int, dict]]:
    """(c) Take each Z ancilla's q label off its vertical coupler, to go beside it as in (b)."""
    names = {f"q{q}": q for q in layout.z_ancillas.values()}
    lifted = []
    for text in [t for t in ax.texts if t.get_text() in names]:
        style = {"fontsize": text.get_fontsize(), "color": text.get_color()}
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
            marker("D", X_RED, INK, 9, "X ancilla and flag", lw=1.5),
            marker("D", X_RED, "white", 9, "X ancilla only"),
            marker("o", Z_BLUE, "white", 10, "Z ancilla"),
            marker("o", "white", GREY, 10, "relay", lw=1.8),
            Line2D([], [], color=X_RED, lw=2.8, label="X-gauge CX"),
            Line2D([], [], color=Z_BLUE, lw=2.8, label="Z-gauge CX"),
            Line2D([], [], color=WEAK_BAND, lw=10, label=f"CZ error above {WEAK_CZ:.0%}"),
            marker("o", "none", WEAK_RING, 15, "poor readout or $T_2$", 1.6),
            Line2D([], [], color=OTHER_REP, lw=1.8, ls=":", label="pair 2, in (a)"),
        ],
        loc="upper center",
        ncol=3,
        frameon=False,
        fontsize=MIN_PT,
        handlelength=2.2,
        columnspacing=1.6,
        bbox_to_anchor=((box.x0 + box.x1) / 2, box.y0 - 0.01),
    )


def box_later_reps(ax: Axes, device: DeviceMap, plan: list) -> None:
    """(a) Box each patch of the reps after the first and label it in one column right of the
    boxes, the distance-5 patch level with its top row and the distance-3 patch with its bottom
    row, each joined to its box by a leader line."""
    boxes = []
    for rep, jobs in enumerate(plan[1:], start=2):
        for job in jobs:
            for d, spot in job.items():
                layout = build_layout(
                    d, device.coords, device.edges, origin=spot.origin, direction=spot.direction
                )
                boxes.append((rep, d, bounds(device.pos, layout.physical_qubits)))
    if not boxes:
        return
    pad = 0.45  # inside rep 1's boxes, so the two stay apart
    column = max(x1 for *_, (_, x1, _, _) in boxes) + 1.0
    for rep, d, (x0, x1, y0, y1) in boxes:
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
        y = y0 + 0.6 if d == 5 else y1 - 0.2
        ax.annotate(
            f"pair {rep}, d={d}",
            xy=(x1 + pad, y),
            xytext=(column, y),
            fontsize=9,
            fontweight="bold",
            color=OTHER_REP,
            va="center",
            bbox=UNDER,
            zorder=8,
            arrowprops={"arrowstyle": "-", "color": OTHER_REP, "lw": 0.9, "shrinkA": 0},
        )


def build_figure(device: DeviceMap, calibration: Calibration, weak_qubits=()) -> Figure:
    plan = day_plan(device.coords, device.edges, calibration)
    if not plan:
        raise ValueError("no usable placement for the d=5 patch: no rep to draw")
    (first, *_) = plan[0]
    spots = first if len(plan[0]) == 1 else {**plan[0][1], **plan[0][0]}
    d3, d5 = (
        build_layout(
            d, device.coords, device.edges, origin=spots[d].origin, direction=spots[d].direction
        )
        for d in (3, 5)
    )
    d3_roles, d5_roles = patch_roles(d3), patch_roles(d5)

    fig = plt.figure(figsize=(WIDTH, HEIGHT))
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
    tidy_d3_labels(ax_d3, device, d3_roles)
    draw_d5_device(ax_d5, device, d5, d5_roles)
    keep_corner_data_labels(ax_d5, 5)
    for ax, layout, size in ((ax_d3, d3, 130), (ax_d5, d5, 95)):  # draw_blueprint's sizes
        mark_flags(ax, device, layout, size)
    for text in ax_d3.texts:
        if text.get_text().isdigit():
            text.set_text(f"q{text.get_text()}")
    for ax in (ax_d3, ax_d5):
        for text in ax.texts:
            if text.get_color() == GREY:
                text.set_color(QUBIT_NUMBER)
            if text.get_text().startswith("q"):
                text.set_bbox(UNDER)
                text.set_zorder(max(text.get_zorder(), 7))
    for ax in (ax_chip, ax_d3, ax_d5):
        for text in ax.texts:
            text.set_fontsize(max(text.get_fontsize(), MIN_PT))
    # Rep r's patches are the paper's pair r; pair r's distance-5 patch is placement r.
    shown = {1: "pair 1", 2: "pair 1 drawn, pair 2 boxed"}.get(
        len(plan), "pair 1 drawn, the others boxed"
    )
    ax_chip.set_title(rf"$\mathbf{{(a)}}$ Fez: {shown}", fontsize=TITLE)
    ax_d3.set_title(
        rf"$\mathbf{{(b)}}$ pair 1, $d = 3$: {len(d3.data)} data qubits, {len(d3.physical_qubits)} in all",
        fontsize=TITLE,
    )
    ax_d5.set_title(
        rf"$\mathbf{{(c)}}$ pair 1, $d = 5$ (placement 1): {len(d5.data)} data qubits, "
        f"{len(d5.physical_qubits)} in all",
        fontsize=TITLE,
    )

    fig.draw_without_rendering()
    draw_key(fig, below=ax_d5)
    taken = taken_boxes(ax_d3)
    mark_weak_couplers(ax_d3, device, d3, calibration, taken)
    mark_weak_qubits(ax_d3, device, d3, weak_qubits, 130)
    lifted = lift_z_labels(ax_d5, d5)
    taken = taken_boxes(ax_d5)
    for q, style in lifted:
        xy = device.pos[q]
        sides = [(xy, 9, 0, "left", "center"), (xy, -9, 0, "right", "center")]
        place(ax_d5, f"q{q}", sides, taken, path_effects=HALO, zorder=7, bbox=UNDER, **style)
    mark_weak_couplers(ax_d5, device, d5, calibration, taken)
    mark_weak_qubits(ax_d5, device, d5, weak_qubits, 95)
    return fig


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Draw the day's reps on ibm_fez.")
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
    parser.add_argument(
        "--weak",
        type=lambda text: [int(q) for q in text.split(",")],
        default=[],
        metavar="Q,Q,...",
        help="qubits to ring as weak (default: none)",
    )
    args = parser.parse_args(argv)
    fig = build_figure(
        load_device_map(FEZ_MAP), Calibration.load(args.calibration), weak_qubits=args.weak
    )
    fig.savefig(args.out, dpi=300)
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
