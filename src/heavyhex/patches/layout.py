"""Embed a heavy-hex patch on a device coupling map.

Data sit on even offsets from the anchor, ancillas and relays on odd ones.
Supports are 1-based code labels; every other value is a physical device id.
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
    wires: tuple[tuple[str, int, int], ...]

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
    first_data_position: tuple[int, int],
) -> HeavyHexLayout:
    """Embed a patch at a specified top-left data site; reject missing bonds."""
    operators = build_operators(distance)
    d = operators.distance
    x0, y0 = first_data_position
    position_to_qubit = {tuple(position): q for q, position in enumerate(coords)}
    bonds = {tuple(sorted(edge)) for edge in edges}

    def site(x: int, y: int) -> int:
        try:
            return position_to_qubit[x, y]
        except KeyError as error:
            raise ValueError(f"no device qubit at {(x, y)}") from error

    data = {
        label(row, column, d): site(x0 + 2 * (column - 1), y0 + 2 * (row - 1))
        for column in range(1, d + 1)
        for row in range(1, d + 1)
    }
    x_ancillas: dict[str, int] = {}
    z_ancillas: dict[str, int] = {}
    relays: list[int] = []
    wires: list[tuple[str, int, int]] = []

    for name, support in operators.x_gauges.items():
        row, column = _row_column(support[0], d)
        ancilla = site(x0 + 2 * column - 1, y0 + 2 * (row - 1))
        x_ancillas[name] = ancilla
        wires.extend(("x", data[q], ancilla) for q in support)

    for name, support in operators.z_gauges.items():
        row, column = _row_column(support[0], d)
        y = y0 + 2 * row - 1
        if len(support) == 4:
            # Block arms run through the in-row X ancillas, which double as flags.
            x = x0 + 2 * column - 1
            ancilla = site(x, y)
            wires.extend(("z", site(x, arm_y), ancilla) for arm_y in (y - 1, y + 1))
        else:
            # Boundary pairs have no in-row flag to borrow: they reach data through relays.
            x = x0 - 1 if column == 1 else x0 + 2 * (d - 1) + 1
            ancilla = site(x, y)
            for q, arm_y in zip(support, (y - 1, y + 1)):
                relay = site(x, arm_y)
                relays.append(relay)
                wires.extend((("z", data[q], relay), ("z", relay, ancilla)))
        z_ancillas[name] = ancilla

    assigned = [*data.values(), *x_ancillas.values(), *z_ancillas.values(), *relays]
    if len(set(assigned)) != len(assigned):
        raise ValueError("device qubit assigned to multiple patch roles")
    for _, a, b in wires:
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
        wires=tuple(wires),
    )


def _row_column(code_label: int, distance: int) -> tuple[int, int]:
    """Inverse of label(): the 1-based (row, column) of a code qubit."""
    return (code_label - 1) % distance + 1, (code_label - 1) // distance + 1
