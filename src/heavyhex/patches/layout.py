"""Place a heavy-hex patch on the chip's coupling map.

Data qubits sit two sites apart, with ancillas and relays on the sites between.
Supports use 1-based code labels; every other number is a physical qubit.
"""

from __future__ import annotations

from dataclasses import dataclass

from .operators import build_operators, label


@dataclass
class HeavyHexLayout:
    distance: int
    data: dict[int, int]
    x_gauges: dict[str, tuple[int, ...]]
    z_gauges: dict[str, tuple[int, ...]]
    x_stabilizers: dict[str, tuple[int, ...]]
    z_stabilizers: dict[str, tuple[int, ...]]
    x_ancillas: dict[str, int]
    z_ancillas: dict[str, int]
    relays: tuple[int, ...]
    couplings: tuple[tuple[str, int, int], ...]

    @property
    def physical_qubits(self) -> set[int]:
        return (
            set(self.data.values())
            | set(self.x_ancillas.values())
            | set(self.z_ancillas.values())
            | set(self.relays)
        )


def build_layout(
    distance: int,
    coords: list[list[int]],
    edges: list[tuple[int, int]],
    *,
    origin: tuple[int, int],
    direction: tuple[int, int] = (1, 1),
) -> HeavyHexLayout:
    """Put data qubit 1 at origin, and fail if the patch needs a bond the chip doesn't have.

    Columns run along x and rows along y, each the way `direction` points (+1 or -1),
    so the patch can face any of four ways.
    """
    if direction not in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
        raise ValueError(f"direction must be (+-1, +-1), got {direction!r}")
    operators = build_operators(distance)
    d = operators.distance
    x0, y0 = origin
    dx, dy = direction
    position_to_qubit = {tuple(position): q for q, position in enumerate(coords)}
    bonds = {tuple(sorted(edge)) for edge in edges}

    def site(x: int, y: int) -> int:
        try:
            return position_to_qubit[x, y]
        except KeyError as error:
            raise ValueError(f"no device qubit at {(x, y)}") from error

    data = {
        label(row, column, d): site(x0 + dx * 2 * (column - 1), y0 + dy * 2 * (row - 1))
        for column in range(1, d + 1)
        for row in range(1, d + 1)
    }
    x_ancillas: dict[str, int] = {}
    z_ancillas: dict[str, int] = {}
    relays: list[int] = []
    couplings: list[tuple[str, int, int]] = []

    for name, support in operators.x_gauges.items():
        row, column = _row_column(support[0], d)
        ancilla = site(x0 + dx * (2 * column - 1), y0 + dy * 2 * (row - 1))
        x_ancillas[name] = ancilla
        couplings.extend(("x", data[q], ancilla) for q in support)

    for name, support in operators.z_gauges.items():
        row, column = _row_column(support[0], d)
        y = y0 + dy * (2 * row - 1)
        if len(support) == 4:
            # A Z4 gauge reaches its data through the X ancillas above and below it,
            # which double as flags.
            x = x0 + dx * (2 * column - 1)
            ancilla = site(x, y)
            couplings.extend(("z", site(x, arm_y), ancilla) for arm_y in (y - dy, y + dy))
        else:
            # Boundary pairs have no X ancilla in their row, so they go through relays.
            x = x0 - dx if column == 1 else x0 + dx * (2 * d - 1)
            ancilla = site(x, y)
            for q, arm_y in zip(support, (y - dy, y + dy)):
                relay = site(x, arm_y)
                relays.append(relay)
                couplings.extend((("z", data[q], relay), ("z", relay, ancilla)))
        z_ancillas[name] = ancilla

    assigned = [*data.values(), *x_ancillas.values(), *z_ancillas.values(), *relays]
    if len(set(assigned)) != len(assigned):
        raise ValueError("device qubit assigned to multiple patch roles")
    for _, a, b in couplings:
        if tuple(sorted((a, b))) not in bonds:
            raise ValueError(f"patch requires a missing device bond: {a}-{b}")
    return HeavyHexLayout(
        distance=d,
        data=data,
        x_gauges=dict(operators.x_gauges),
        z_gauges=dict(operators.z_gauges),
        x_stabilizers=dict(operators.x_stabilizers),
        z_stabilizers=dict(operators.z_stabilizers),
        x_ancillas=x_ancillas,
        z_ancillas=z_ancillas,
        relays=tuple(relays),
        couplings=tuple(couplings),
    )


def _row_column(code_label: int, distance: int) -> tuple[int, int]:
    """Inverse of label(): the 1-based (row, column) of a code qubit."""
    return (code_label - 1) % distance + 1, (code_label - 1) // distance + 1
