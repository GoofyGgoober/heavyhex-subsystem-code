"""The d=3 and d=5 placements on Fez drawn in docs/figures."""

import json
from collections import Counter
from pathlib import Path

import pytest

from heavyhex.patches.layout import build_layout

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"
# Must match D3_ORIGIN / D5_ORIGIN in docs/figures/draw_blueprint.py.
ORIGINS = {3: (5, 1), 5: (3, 7)}


@pytest.fixture(scope="module")
def device_map():
    return json.loads((FIGURES / "fez_map.json").read_text())


@pytest.fixture
def layouts(device_map):
    return {
        distance: build_layout(distance, device_map["coords"], device_map["edges"], origin=origin)
        for distance, origin in ORIGINS.items()
    }


@pytest.mark.parametrize(("distance", "sites", "bonds"), [(3, 23, 24), (5, 65, 72)])
def test_patch_sits_on_real_couplings(layouts, device_map, distance, sites, bonds):
    device_bonds = {tuple(sorted(edge)) for edge in device_map["edges"]}
    patch = layouts[distance]
    d = patch.distance
    assert d == distance
    assert len(patch.data) == d * d
    assert len(patch.x_ancillas) == len(patch.x_gauges) == d * (d - 1)
    assert len(patch.z_ancillas) == len(patch.z_gauges) == (d * d - 1) // 2
    assert len(patch.relays) == 2 * (d - 1)
    assert len(patch.physical_qubits) == sites
    patch_bonds = {tuple(sorted((a, b))) for _, a, b in patch.couplings}
    assert len(patch.couplings) == len(patch_bonds) == bonds
    assert patch_bonds <= device_bonds
    assert {q for edge in patch_bonds for q in edge} == patch.physical_qubits
    # Heavy-hex is degree <= 3; a busier site would not be routable.
    assert max(Counter(q for edge in patch_bonds for q in edge).values()) <= 3


def test_patches_share_no_qubits_or_direct_couplings(layouts, device_map):
    small, large = (layouts[d].physical_qubits for d in (3, 5))
    assert small.isdisjoint(large)
    assert len(small | large) == 88
    assert not any(
        (a in small and b in large) or (a in large and b in small) for a, b in device_map["edges"]
    )


def test_d3_matches_the_original_falcon_mapping(layouts):
    """The d=3 placement still reproduces Sundaresan et al. Fig. 4a on fez."""
    mapping = {
        int(q): fez for q, fez in json.loads((FIGURES / "d3_fez_layout.json").read_text()).items()
    }
    patch = layouts[3]
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


def test_missing_physical_bond_is_rejected(layouts, device_map):
    _, a, b = layouts[5].couplings[0]
    edges = [edge for edge in device_map["edges"] if set(edge) != {a, b}]
    with pytest.raises(ValueError, match="missing device bond"):
        build_layout(5, device_map["coords"], edges, origin=ORIGINS[5])


def test_off_chip_placement_is_rejected(device_map):
    with pytest.raises(ValueError, match="no device qubit"):
        build_layout(5, device_map["coords"], device_map["edges"], origin=(3, 11))


def test_saved_d5_manifest_matches_generated_layout(layouts):
    saved = json.loads((FIGURES / "d5_fez_layout.json").read_text())
    patch = layouts[5]
    assert saved["first_data_position"] == list(ORIGINS[5])
    assert {int(q): physical for q, physical in saved["data_qubits"].items()} == patch.data
    assert saved["x_gauge_ancillas"] == patch.x_ancillas
    assert saved["z_gauge_ancillas"] == patch.z_ancillas
    assert set(saved["boundary_relays"]) == set(patch.relays)
    assert {tuple(coupling) for coupling in saved["couplings"]} == set(patch.couplings)
    for field in ("x_gauges", "z_gauges", "x_stabilizers", "z_stabilizers"):
        assert {name: tuple(support) for name, support in saved[field].items()} == getattr(
            patch, field
        )
