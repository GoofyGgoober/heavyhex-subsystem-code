"""Picking patch spots from the last Fez calibration, and pulling a fresh one when it's old."""

import json
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from heavyhex.circuits.flagged import memory_circuit_flagged, qubit_roles
from heavyhex.patches.layout import build_layout
from heavyhex.patches.operators import build_operators
from heavyhex.patches.placement import (
    Calibration,
    best_spots,
    chip_qubits,
    clean_places,
    cost,
    day_plan,
    fez_qubits,
    neighbourhood,
    problems,
    spots,
    todays_calibration,
    weak,
)

FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"
DEVICE = json.loads((FIGURES / "fez_map.json").read_text())
COORDS, EDGES = DEVICE["coords"], DEVICE["edges"]
SAVED = Calibration.load(FIGURES / "fez_calibration.json")
# IBM's calibration of 3 October 2026, 17:05 UTC: two clean places for the d=5 patch, and
# none for the d=3 patch beside the second. Kept fixed, unlike the day's calibration.
OCTOBER_3 = Calibration.load(Path(__file__).parent / "data" / "fez_calibration_2026-10-03.json")
# Every part working and alike, so every place is clean.
ALL_GOOD = replace(
    SAVED,
    cz=dict.fromkeys(SAVED.cz, 0.003),
    readout=dict.fromkeys(SAVED.readout, 0.01),
    t1_us=dict.fromkeys(SAVED.t1_us, 150.0),
    t2_us=dict.fromkeys(SAVED.t2_us, 100.0),
    single_qubit=None,
    broken_qubits=(),
)


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


def test_fetch_keeps_the_device_durations(tmp_path, monkeypatch):
    runtime = pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    class Service:
        def backend(self, name):
            return FakeKingston()

    monkeypatch.setattr(runtime, "QiskitRuntimeService", Service)
    fetched = Calibration.fetch("ibm_kingston")
    assert fetched.durations_ns == {"x": 32, "cz": 68, "measure": 2280, "reset": 2312}
    assert fetched.single_qubit[0] == FakeKingston().target["sx"][(0,)].error
    fetched.save(tmp_path / "calibration.json")
    assert Calibration.load(tmp_path / "calibration.json") == fetched


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
    broken = replace(SAVED, cz=dict.fromkeys(SAVED.cz, 1.0))  # no spot without a broken part
    with pytest.raises(ValueError, match="would use broken parts: coupler"):
        fez_qubits(5, broken)


def test_cost_charges_parts_as_often_as_a_round_uses_them():
    layout = next(iter(spots(3, COORDS, EDGES).values()))
    flags = {flag for arms in qubit_roles(build_operators(3)).z4_arms.values() for _, flag in arms}

    def worse_readout(q):
        worse = replace(SAVED, readout={**SAVED.readout, q: SAVED.readout[q] + 0.01})
        return cost(layout, worse) - cost(layout, SAVED)

    assert worse_readout(layout.data[1]) == pytest.approx(0)  # read once, at the end
    for name, q in layout.x_ancillas.items():
        # Reset and measured in the X half, and again in the Z half if it is a flag.
        assert worse_readout(q) == pytest.approx(0.04 if name in flags else 0.02)


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


def test_equal_costs_tie_the_same_way_however_they_were_summed(monkeypatch):
    import heavyhex.patches.placement as placement

    exact = best_spots(COORDS, EDGES, SAVED)
    noise = iter(range(10**6))
    monkeypatch.setattr(
        placement, "cost", lambda layout, calibration: 1.0 + next(noise) % 3 * 1e-15
    )
    monkeypatch.setattr(placement, "problems", lambda layout, calibration: [])
    first = best_spots(COORDS, EDGES, SAVED)
    assert best_spots(COORDS, EDGES, SAVED) == first
    assert exact  # the real costs still give a placement


def test_a_qubit_ibm_cannot_drive_counts_as_broken():
    from heavyhex.patches.placement import broken_qubit

    assert not broken_qubit(4, replace(SAVED, single_qubit={4: 3e-4}))
    assert broken_qubit(4, replace(SAVED, single_qubit={4: 1.0}))


def test_a_past_calibration_comes_from_ibms_history():
    pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    device = FakeKingston()

    class Then:
        name, num_qubits, target = "ibm_kingston", device.num_qubits, device.target

        def properties(self, datetime=None):
            assert datetime is not None
            return device.properties()

    now = Calibration.from_device(device)
    then = Calibration.from_device(Then(), at=datetime.now())
    assert then.readout == now.readout and then.single_qubit == pytest.approx(now.single_qubit)
    assert then.cz.keys() == now.cz.keys()


def layout_at(distance, spot):
    return build_layout(distance, COORDS, EDGES, origin=spot.origin, direction=spot.direction)


def test_a_place_is_listed_once_at_its_best_spot():
    places = clean_places(3, COORDS, EDGES, ALL_GOOD)
    footprints = [frozenset(layout.physical_qubits) for _, layout in places]
    assert len(footprints) == len(set(footprints)) == 30
    costs = [cost(layout, ALL_GOOD) for _, layout in places]
    assert costs == sorted(costs)


def test_the_day_has_a_rep_for_each_clean_d5_place_up_to_two():
    plan = day_plan(COORDS, EDGES, OCTOBER_3)
    assert len(plan) == 2 == len(clean_places(5, COORDS, EDGES, OCTOBER_3))
    first, second = plan
    assert first == [best_spots(COORDS, EDGES, OCTOBER_3)]  # the usual pair, in one job
    # No clean d=3 place misses the second d=5 place, so d=3 runs alone straight after.
    assert [set(job) for job in second] == [{5}, {3}]
    assert day_plan(COORDS, EDGES, OCTOBER_3, reps=1) == [first]


@pytest.mark.parametrize("calibration", [ALL_GOOD, OCTOBER_3], ids=["all good", "3 October"])
def test_each_rep_puts_d3_on_a_place_of_its_own_beside_d5_when_it_can(calibration):
    used = []
    plan = day_plan(COORDS, EDGES, calibration)
    assert len(plan) == 2
    for jobs in plan:
        five = layout_at(5, jobs[0][5])
        three = layout_at(3, jobs[0][3] if len(jobs) == 1 else jobs[1][3])
        place = frozenset(three.physical_qubits)
        assert place not in used
        beside = [
            layout
            for _, layout in clean_places(3, COORDS, EDGES, calibration)
            if frozenset(layout.physical_qubits) not in used
            and not layout.physical_qubits & neighbourhood(five, EDGES)
        ]
        if len(jobs) == 1:  # in one job: it doesn't touch d=5
            assert not three.physical_qubits & neighbourhood(five, EDGES)
            assert three.physical_qubits == beside[0].physical_qubits
        else:  # alone, straight after: only when nothing fits beside
            assert not beside and [set(job) for job in jobs] == [{5}, {3}]
        used.append(place)


def test_no_clean_d5_place_means_no_rep():
    broken = replace(ALL_GOOD, broken_qubits=tuple(ALL_GOOD.t1_us)[::4])
    assert not clean_places(5, COORDS, EDGES, broken) or day_plan(COORDS, EDGES, broken)
    nothing = replace(ALL_GOOD, cz=dict.fromkeys(ALL_GOOD.cz, 1.0))
    with pytest.raises(ValueError, match="no clean place for the d=3 patch"):
        day_plan(COORDS, EDGES, nothing)


def test_a_given_spot_gets_its_own_qubits():
    (job,) = day_plan(COORDS, EDGES, OCTOBER_3)[0]
    assert fez_qubits(5, OCTOBER_3) == fez_qubits(5, OCTOBER_3, job[5])
    alone = day_plan(COORDS, EDGES, OCTOBER_3)[1][0][5]
    assert fez_qubits(5, OCTOBER_3, alone) == chip_qubits(layout_at(5, alone))
