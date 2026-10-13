"""The follow-up's placement on ibm_kingston: the thresholds, the score S, and the d=3
patch nested in the d=5 footprint."""

from dataclasses import replace
from pathlib import Path

import pytest

from followup.placement import (
    MAX_CZ,
    MAX_READOUT,
    MIN_DATA_T2_US,
    MIN_T1_US,
    MIN_T2_US,
    ROUND_US,
    by_role,
    chip_map,
    fetch_calibration,
    place,
    rank,
    score,
    shortfalls,
)
from heavyhex.patches.layout import build_layout
from heavyhex.patches.placement import Calibration, Spot

# IBM's calibration of ibm_kingston on 13 October 2026, from which the plan was made.
OCTOBER_13 = Calibration.load(
    Path(__file__).parent / "data" / "kingston_calibration_2026-10-13.json"
)
# Every part working, alike and within the thresholds.
GOOD = replace(
    OCTOBER_13,
    cz=dict.fromkeys(OCTOBER_13.cz, 0.003),
    readout=dict.fromkeys(OCTOBER_13.readout, 0.01),
    t1_us=dict.fromkeys(OCTOBER_13.t1_us, 200.0),
    t2_us=dict.fromkeys(OCTOBER_13.t2_us, 150.0),
    single_qubit=None,
    broken_qubits=(),
)
PLAN_D5 = Spot((7, 9), (1, -1))
PLAN_D3 = Spot((11, 9), (1, -1))


def layout_at(distance, spot):
    coords, edges = chip_map()
    return build_layout(distance, coords, edges, origin=spot.origin, direction=spot.direction)


def changed(calibration, **parts):
    """The calibration with some of its numbers replaced, as {field: {part: value}}."""
    return replace(
        calibration,
        **{field: {**getattr(calibration, field), **new} for field, new in parts.items()},
    )


@pytest.fixture(scope="module")
def october_13():
    return place(OCTOBER_13)


def test_the_plans_placement_on_13_october(october_13):
    five, three = october_13[5], october_13[3]
    assert (five.spot, three.spot) == (PLAN_D5, PLAN_D3)
    assert len(five.layout.physical_qubits) == 65
    assert (min(five.layout.physical_qubits), max(five.layout.physical_qubits)) == (6, 94)
    assert len(three.layout.physical_qubits) == 23
    assert three.layout.physical_qubits <= five.layout.physical_qubits
    assert all(50 <= q <= 94 for q in three.layout.physical_qubits)
    assert not five.usable and not three.usable  # nothing is within the thresholds today


def test_a_footprints_two_orientations_tie_and_the_spots_order_decides():
    first, second = rank(5, OCTOBER_13)[:2]
    assert first.layout.physical_qubits == second.layout.physical_qubits
    assert first.score == pytest.approx(second.score, abs=1e-12)
    assert first.spot < second.spot


def test_every_part_beyond_the_thresholds_is_named(october_13):
    five = [str(part) for part in october_13[5].shortfalls]
    assert len(five) == 24
    assert {
        "data 86: T2 24 us",
        "data 88: T2 19 us",
        "data 48: T2 21 us",
        "data 10: T2 25 us",
        "coupler 85-86: CZ 4.7%",
        "coupler 86-87: CZ 5.7%",
    } <= set(five)
    assert [str(part) for part in october_13[3].shortfalls] == [
        "data 52: T2 56 us",
        "data 52: readout 3.3%",
        "data 54: readout 4.5%",
        "data 72: readout 5.2%",
        "Z ancilla 78: T2 32 us",
        "Z ancilla 78: readout 4.4%",
    ]


def test_the_qubits_in_each_role(october_13):
    assert october_13[3].roles == {
        "data": [50, 52, 54, 70, 72, 74, 90, 92, 94],
        "x_ancillas": [51, 53, 71, 73, 91, 93],
        "z_ancillas": [58, 59, 78, 79],
        "flags": [51, 71, 73, 93],
        "relays": [55, 69, 75, 89],
    }
    five = october_13[5].roles
    assert [len(qubits) for qubits in five.values()] == [25, 20, 12, 16, 8]
    assert set(five["flags"]) <= set(five["x_ancillas"])
    assert {69, 89} <= set(five["flags"])  # relays at d=3
    assert sorted(october_13[5].qubits) == sorted(october_13[5].layout.physical_qubits)


def test_every_d5_footprint_holds_data_qubits_86_and_88():
    placements = rank(5, GOOD)
    assert len({frozenset(p.layout.physical_qubits) for p in placements}) == 12
    assert all({86, 88} <= set(p.roles["data"]) for p in placements)


def test_the_score_charges_the_data_their_t2_and_no_resets():
    layout = layout_at(3, PLAN_D3)
    base = score(layout, GOOD)
    data, ancilla = layout.data[1], next(iter(layout.z_ancillas.values()))
    t = ROUND_US[3]
    assert score(layout, changed(GOOD, t2_us={data: 50.0})) - base == pytest.approx(
        t / 100 - t / 300
    )
    assert score(layout, changed(GOOD, t2_us={ancilla: 50.0})) == base
    assert score(layout, replace(GOOD, t1_us=dict.fromkeys(GOOD.t1_us, 60.0))) == base

    def worse_readout(q):
        return score(layout, changed(GOOD, readout={q: 0.02})) - base

    flags = by_role(layout)["flags"]
    assert worse_readout(data) == pytest.approx(0)  # read once, at the end
    for q in layout.x_ancillas.values():
        # Measured in the X half, and again in the Z half if it is a flag; never reset.
        assert worse_readout(q) == pytest.approx(0.02 if q in flags else 0.01)


def test_the_thresholds_include_their_limits():
    layout = layout_at(3, PLAN_D3)
    data, ancilla = layout.data[1], next(iter(layout.z_ancillas.values()))
    pair = tuple(sorted(layout.couplings[0][1:]))

    def beyond(**parts):
        return [(s.part, s.quantity) for s in shortfalls(layout, changed(GOOD, **parts))]

    at_the_limits = {
        "t1_us": {data: MIN_T1_US},
        "t2_us": {data: MIN_DATA_T2_US, ancilla: MIN_T2_US},
        "readout": {data: MAX_READOUT},
        "cz": {pair: MAX_CZ},
    }
    assert beyond(**at_the_limits) == []
    assert beyond(t2_us={data: 79.9}) == [(data, "T2")]
    assert beyond(t2_us={ancilla: 60.0}) == []  # only data qubits need 80 us
    assert beyond(t2_us={ancilla: 49.9}) == [(ancilla, "T2")]
    assert beyond(t1_us={ancilla: 79.9}) == [(ancilla, "T1")]
    assert beyond(readout={ancilla: 0.0301}) == [(ancilla, "readout")]
    assert beyond(cz={pair: 0.0101}) == [(pair, "CZ")]


def test_a_spot_within_the_thresholds_beats_any_with_a_lower_score():
    favoured = layout_at(5, PLAN_D5)
    sharper = changed(GOOD, t2_us=dict.fromkeys(favoured.data.values(), 300.0))
    assert place(sharper)[5].spot == PLAN_D5  # its data dephase least

    slow = changed(sharper, t1_us={6: 60.0})  # one of its data qubits
    chosen = place(slow)[5]
    assert chosen.usable and 6 not in chosen.layout.physical_qubits
    assert chosen.score > score(favoured, slow)

    # With nothing within the thresholds the lowest score wins, and every part is named.
    nowhere = replace(sharper, t1_us=dict.fromkeys(GOOD.t1_us, 60.0))
    lowest = place(nowhere)[5]
    assert lowest.spot == PLAN_D5
    assert [s.quantity for s in lowest.shortfalls] == ["T1"] * 65


def test_the_d3_patch_is_the_lowest_score_inside_the_d5_footprint(october_13):
    footprint = october_13[5].layout.physical_qubits
    inside = rank(3, OCTOBER_13, inside=footprint)
    assert len(inside) == 18
    assert all(p.layout.physical_qubits <= footprint for p in inside)
    assert october_13[3] == inside[0]
    # Nothing inside is within the thresholds, so the lowest score wins over the fewest
    # shortfalls, which would put data qubit 48 (T2 21 us) in the patch.
    fewest = min(inside, key=lambda p: len(p.shortfalls))
    assert ("data", 48) in {(s.role, s.part) for s in fewest.shortfalls}
    assert fewest.score > october_13[3].score


def test_a_broken_part_is_never_used():
    cut = changed(GOOD, cz={(6, 7): 1.0})  # used by the plan's d=5 footprint
    for p in rank(5, cut):
        assert (6, 7) not in {tuple(sorted((a, b))) for _, a, b in p.layout.couplings}
    assert place(cut)[5].spot != PLAN_D5
    with pytest.raises(ValueError, match="every d=5 spot"):
        place(replace(GOOD, broken_qubits=(86,)))  # in every d=5 footprint


def test_fetch_saves_the_calibration_and_checks_the_map(tmp_path, monkeypatch):
    runtime = pytest.importorskip("qiskit_ibm_runtime")
    from qiskit_ibm_runtime.fake_provider import FakeKingston, FakeTorino

    devices = {"ibm_kingston": FakeKingston, "ibm_torino": FakeTorino}

    class Service:
        def backend(self, name):
            return devices[name]()

    monkeypatch.setattr(runtime, "QiskitRuntimeService", Service)
    fetched = fetch_calibration(tmp_path / "kingston.json")
    assert Calibration.load(tmp_path / "kingston.json") == fetched
    with pytest.raises(ValueError, match="couplers differ"):
        fetch_calibration(tmp_path / "torino.json", "ibm_torino")
    assert not (tmp_path / "torino.json").exists()
