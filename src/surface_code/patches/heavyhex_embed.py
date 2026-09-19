"""Relay-preserving heavy-hex embeddings on a device coupling map.

Takes the device-independent operators from heavyhex.py and assigns every
role to a physical qubit, rejecting missing bonds. Supports stay 1-based
code-qubit labels; data/ancilla/relay values are physical device ids.
"""

from __future__ import annotations

from dataclasses import dataclass

from .heavyhex import build_operators, label


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


def operator_name(pauli: str, support: tuple[int, ...]) -> str:
    return "".join(f"{pauli}{qubit}" for qubit in support)


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

    def block(row: int, column: int) -> tuple[int, int, int, int]:
        return (
            label(row, column, d),
            label(row + 1, column, d),
            label(row, column + 1, d),
            label(row + 1, column + 1, d),
        )

    data = {
        label(row, column, d): site(x0 + 2 * (column - 1), y0 + 2 * (row - 1))
        for column in range(1, d + 1)
        for row in range(1, d + 1)
    }
    x_gauges, z_gauges, x_ancillas, z_ancillas = {}, {}, {}, {}
    x_stabilizers, z_stabilizers = {}, {}
    relays, wires = [], []
    for column in range(1, d):
        for row in range(1, d + 1):
            support = (label(row, column, d), label(row, column + 1, d))
            name = operator_name("X", support)
            assert operators.x_gauges[name] == support
            ancilla = site(x0 + 2 * column - 1, y0 + 2 * (row - 1))
            x_gauges[name] = support
            x_ancillas[name] = ancilla
            wires.extend(("x", data[q], ancilla) for q in support)

    for row in range(1, d):
        y = y0 + 2 * row - 1
        for column in range(1, d):
            support = block(row, column)
            if (row + column) % 2 == 0:
                name = operator_name("X", support)
                assert operators.x_stabilizers[name] == support
                x_stabilizers[name] = support
                continue
            name = operator_name("Z", support)
            assert operators.z_gauges[name] == support
            x = x0 + 2 * column - 1
            ancilla = site(x, y)
            z_gauges[name] = support
            z_ancillas[name] = ancilla
            # Block arms run through the in-row X ancillas, which double as flags.
            wires.extend(("z", site(x, arm_y), ancilla) for arm_y in (y - 1, y + 1))

        # Alternate left and right boundary gauges, as in the d=3 figure.
        column = 1 if row % 2 else d
        x = x0 - 1 if row % 2 else x0 + 2 * (d - 1) + 1
        support = (label(row, column, d), label(row + 1, column, d))
        name = operator_name("Z", support)
        assert operators.z_gauges[name] == support
        ancilla = site(x, y)
        z_gauges[name] = support
        z_ancillas[name] = ancilla
        for q, arm_y in zip(support, (y - 1, y + 1)):
            relay = site(x, arm_y)
            relays.append(relay)
            wires.extend((("z", data[q], relay), ("z", relay, ancilla)))

        support = tuple(label(r, c, d) for c in range(1, d + 1) for r in (row, row + 1))
        name = operator_name("Z", support)
        assert operators.z_stabilizers[name] == support
        z_stabilizers[name] = support

    for column in range(1, d):
        row = d if column % 2 else 1
        support = (label(row, column, d), label(row, column + 1, d))
        name = operator_name("X", support)
        assert operators.x_stabilizers[name] == support
        x_stabilizers[name] = support

    assert x_gauges == operators.x_gauges
    assert z_gauges == operators.z_gauges
    assert x_stabilizers == operators.x_stabilizers
    assert z_stabilizers == operators.z_stabilizers
    assigned = [*data.values(), *x_ancillas.values(), *z_ancillas.values(), *relays]
    if len(set(assigned)) != len(assigned):
        raise ValueError("device qubit assigned to multiple patch roles")
    for _, a, b in wires:
        if tuple(sorted((a, b))) not in bonds:
            raise ValueError(f"patch requires a missing device bond: {a}-{b}")
    return HeavyHexLayout(
        d,
        data,
        x_gauges,
        z_gauges,
        x_stabilizers,
        z_stabilizers,
        x_ancillas,
        z_ancillas,
        tuple(relays),
        tuple(wires),
    )
