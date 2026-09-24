"""Picking patch spots from the last Fez calibration, and pulling a fresh one when it's old."""

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from heavyhex.circuits.flagged import memory_circuit_flagged
from heavyhex.patches.layout import build_layout
from heavyhex.patches.operators import build_operators
from heavyhex.patches.placement import (
    Calibration,
    best_spots,
    fez_qubits,
    problems,
    spots,
    todays_calibration,
    weak,
)

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"
DEVICE = json.loads((FIGURES / "fez_map.json").read_text())
COORDS, EDGES = DEVICE["coords"], DEVICE["edges"]
SAVED = Calibration.load(FIGURES / "fez_calibration.json")


def pulled(days_ago: int) -> str:
    return (datetime.now().astimezone() - timedelta(days=days_ago)).isoformat(timespec="seconds")


def test_todays_calibration_is_reused(tmp_path, monkeypatch):
    replace(SAVED, pulled=pulled(0)).save(tmp_path / "calibration.json")
    monkeypatch.setattr(Calibration, "fetch", classmethod(lambda cls, backend: 1 / 0))
    assert todays_calibration(path=tmp_path / "calibration.json").pulled == pulled(0)


@pytest.mark.parametrize("days_ago", [1, None])
def test_old_or_missing_calibration_is_pulled_again(tmp_path, monkeypatch, days_ago):
    path = tmp_path / "calibration.json"
    if days_ago is not None:
        replace(SAVED, pulled=pulled(days_ago)).save(path)
    fresh = replace(SAVED, pulled=pulled(0))
    monkeypatch.setattr(Calibration, "fetch", classmethod(lambda cls, backend: fresh))
    assert todays_calibration(path=path) == fresh
    assert Calibration.load(path) == fresh


@pytest.mark.filterwarnings("ignore:.*weak parts")
@pytest.mark.parametrize("distance", [3, 5])
def test_every_cx_lands_on_a_fez_bond(distance):
    pytest.importorskip("qiskit")
    working = replace(SAVED, cz={pair: 0.003 for pair in SAVED.cz})
    qubits = fez_qubits(distance, working)
    circuit, _ = memory_circuit_flagged(build_operators(distance), rounds=2)
    assert len(set(qubits)) == len(qubits) == circuit.num_qubits
    bonds = {frozenset(edge) for edge in EDGES}
    for instruction in circuit.data:
        if instruction.operation.name == "cx":
            pair = frozenset(qubits[circuit.find_bit(q).index] for q in instruction.qubits)
            assert pair in bonds


def test_weak_parts_are_named_and_warned_about():
    spot = best_spots(COORDS, EDGES, SAVED)[3]
    layout = build_layout(3, COORDS, EDGES, origin=spot.origin, direction=spot.direction)
    pair = tuple(sorted(layout.couplings[0][1:]))
    shaky = replace(SAVED, cz={**SAVED.cz, pair: 0.04})
    assert f"coupler {pair[0]}-{pair[1]}: CZ error 4.0%" in weak(layout, shaky)
    with pytest.warns(UserWarning, match="weak parts"):
        fez_qubits(3, SAVED)


def test_placement_with_broken_parts_is_refused():
    with pytest.raises(ValueError, match="coupler 27-28 is broken"):
        fez_qubits(5, SAVED)


def test_picks_have_as_few_broken_parts_as_possible():
    picked = best_spots(COORDS, EDGES, SAVED)
    for d, spot in picked.items():
        layout = build_layout(d, COORDS, EDGES, origin=spot.origin, direction=spot.direction)
        fewest = min(len(problems(other, SAVED)) for other in spots(d, COORDS, EDGES).values())
        assert len(problems(layout, SAVED)) == fewest, d


def test_patches_fit_facing_all_four_ways():
    for d in (3, 5):
        assert {spot.direction for spot in spots(d, COORDS, EDGES)} == {
            (1, 1),
            (1, -1),
            (-1, 1),
            (-1, -1),
        }


def test_calibration_survives_a_save_and_load(tmp_path):
    SAVED.save(tmp_path / "calibration.json")
    assert Calibration.load(tmp_path / "calibration.json") == SAVED


@pytest.mark.parametrize("direction", [(0, 1), (2, 1), (1,)])
def test_bad_direction_is_rejected(direction):
    with pytest.raises(ValueError, match="direction"):
        build_layout(3, COORDS, EDGES, origin=(5, 1), direction=direction)
