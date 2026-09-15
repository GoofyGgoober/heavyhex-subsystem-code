"""Offline checks for the documentation's two heavy-hex device layouts."""

import importlib
import json
from collections import Counter
from pathlib import Path

import pytest

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"


@pytest.fixture
def layout_builder(monkeypatch):
    monkeypatch.syspath_prepend(str(FIGURES))
    return importlib.import_module("heavyhex_layout").build_layout


@pytest.fixture
def device_map():
    return json.loads((FIGURES / "fez_map.json").read_text())


@pytest.fixture
def layouts(layout_builder, device_map):
    return [
        layout_builder(d, device_map["coords"], device_map["edges"], first_data_position=origin)
        for d, origin in ((3, (5, 1)), (5, (3, 7)))
    ]


def binary_rank(supports):
    pivots = {}
    for support in supports:
        vector = sum(1 << (q - 1) for q in support)
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = vector
                break
            vector ^= pivots[pivot]
    return len(pivots)


def test_patch_roles_counts_and_real_couplings(layouts, device_map):
    bonds = {tuple(sorted(edge)) for edge in device_map["edges"]}
    for patch, expected_qubits, expected_bonds in zip(layouts, (23, 65), (24, 72)):
        d = patch.distance
        assert len(patch.data) == d * d
        assert len(patch.x_ancillas) == len(patch.x_gauges) == d * (d - 1)
        assert len(patch.z_ancillas) == len(patch.z_gauges) == (d * d - 1) // 2
        assert len(patch.relays) == 2 * (d - 1)
        assert len(patch.physical_qubits) == expected_qubits
        patch_bonds = {tuple(sorted((a, b))) for _, a, b in patch.wires}
        assert len(patch.wires) == len(patch_bonds) == expected_bonds
        assert patch_bonds <= bonds
        assert {q for edge in patch_bonds for q in edge} == patch.physical_qubits
        assert max(Counter(q for edge in patch_bonds for q in edge).values()) <= 3


def test_patches_share_no_qubits_or_direct_couplings(layouts, device_map):
    small, large = (patch.physical_qubits for patch in layouts)
    assert small.isdisjoint(large)
    assert len(small | large) == 88
    assert not any(
        (a in small and b in large) or (a in large and b in small) for a, b in device_map["edges"]
    )


def test_d3_matches_the_original_falcon_mapping(layouts):
    mapping = {
        int(q): fez for q, fez in json.loads((FIGURES / "d3_fez_layout.json").read_text()).items()
    }
    patch = layouts[0]
    assert patch.data == {
        k: mapping[q] for k, q in enumerate((2, 10, 17, 5, 13, 21, 9, 16, 24), start=1)
    }
    assert patch.x_ancillas == {
        name: mapping[q]
        for name, q in zip(("X1X4", "X2X5", "X3X6", "X4X7", "X5X8", "X6X9"), (3, 12, 18, 8, 14, 23))
    }
    assert patch.z_ancillas == {
        name: mapping[q]
        for name, q in (("Z1Z2", 4), ("Z4Z5Z7Z8", 11), ("Z2Z3Z5Z6", 15), ("Z8Z9", 22))
    }
    assert set(patch.relays) == {mapping[q] for q in (1, 7, 19, 25)}
    assert set(patch.x_stabilizers) == {"X1X2X4X5", "X4X7", "X3X6", "X5X6X8X9"}
    assert set(patch.z_stabilizers) == {"Z1Z2Z4Z5Z7Z8", "Z2Z3Z5Z6Z8Z9"}


def test_stabilizers_are_central_gauge_products_with_one_logical_qubit(layouts):
    for patch in layouts:
        d = patch.distance
        gx, gz = list(patch.x_gauges.values()), list(patch.z_gauges.values())
        sx, sz = list(patch.x_stabilizers.values()), list(patch.z_stabilizers.values())
        for same_basis, stabilizers, other_basis in ((gx, sx, gz), (gz, sz, gx)):
            assert binary_rank(stabilizers) == len(stabilizers)
            assert binary_rank(same_basis + stabilizers) == binary_rank(same_basis)
            assert all(len(set(s) & set(g)) % 2 == 0 for s in stabilizers for g in other_basis)
        stabilizer_rank = binary_rank(sx) + binary_rank(sz)
        gauge_rank = binary_rank(gx) + binary_rank(gz)
        gauge_qubits = (gauge_rank - stabilizer_rank) // 2
        assert gauge_qubits == (d - 1) ** 2 // 2
        assert d * d - stabilizer_rank - gauge_qubits == 1
        logical_x = tuple(range(1, d + 1))
        logical_z = tuple(range(1, d * d + 1, d))
        for logical, same_basis, other_basis in ((logical_x, gx, gz), (logical_z, gz, gx)):
            assert all(len(set(logical) & set(g)) % 2 == 0 for g in other_basis)
            assert binary_rank(same_basis + [logical]) == binary_rank(same_basis) + 1
        assert len(set(logical_x) & set(logical_z)) % 2 == 1


@pytest.mark.parametrize("distance", [True, 2, 4, 5.0])
def test_invalid_distance_is_rejected(layout_builder, device_map, distance):
    with pytest.raises(ValueError, match="odd integer"):
        layout_builder(
            distance, device_map["coords"], device_map["edges"], first_data_position=(3, 7)
        )


def test_missing_physical_bond_is_rejected(layout_builder, layouts, device_map):
    _, a, b = layouts[1].wires[0]
    edges = [edge for edge in device_map["edges"] if set(edge) != {a, b}]
    with pytest.raises(ValueError, match="missing device bond"):
        layout_builder(5, device_map["coords"], edges, first_data_position=(3, 7))


def test_off_chip_placement_is_rejected(layout_builder, device_map):
    with pytest.raises(ValueError, match="no device qubit"):
        layout_builder(5, device_map["coords"], device_map["edges"], first_data_position=(3, 11))


def test_saved_d5_manifest_matches_generated_layout(layouts):
    saved = json.loads((FIGURES / "d5_fez_layout.json").read_text())
    patch = layouts[1]
    assert {int(q): physical for q, physical in saved["data_qubits"].items()} == patch.data
    assert saved["x_gauge_ancillas"] == patch.x_ancillas
    assert saved["z_gauge_ancillas"] == patch.z_ancillas
    assert set(saved["boundary_relays"]) == set(patch.relays)
    assert {tuple(wire) for wire in saved["couplings"]} == set(patch.wires)
    for field in ("x_gauges", "z_gauges", "x_stabilizers", "z_stabilizers"):
        assert {name: tuple(support) for name, support in saved[field].items()} == getattr(
            patch, field
        )
