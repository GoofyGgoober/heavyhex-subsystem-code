"""The follow-up's decoder: capped weights, the single-fault check, and the distances."""

from math import log
from pathlib import Path

import pytest

pytest.importorskip("pymatching")
pytest.importorskip("stim")

import pymatching
import stim

from followup import decoder

OCTOBER_13 = Path(__file__).parent / "data" / "kingston_calibration_2026-10-13.json"


def dem(text: str) -> stim.DetectorErrorModel:
    return stim.DetectorErrorModel(text)


# A fault on two detectors, and two likely boundary edges, one carrying the observable.
DETOUR = """
error({fault}) D0 D1
error(0.3) D0 L0
error(0.3) D1
"""


def test_caps_bound_every_edge():
    model = dem("error(0.3) D0 D1\nerror(0.02) D1 D2 L0\nerror(0.2) D2\ndetector D3")
    for cap in (0.1, 0.05):
        m = decoder.matching(model, cap)
        assert m.num_detectors == 4
        for _, _, edge in m.edges():
            assert edge["error_probability"] <= cap
            assert edge["weight"] >= log((1 - cap) / cap) - 1e-9
    weights = {round(edge["weight"], 9) for _, _, edge in decoder.matching(model, None).edges()}
    assert len(weights) == 1


def test_the_cap_drops_until_every_single_fault_is_corrected():
    # Uncapped, the boundary detour (two edges at 0.3) beats each single fault.
    assert decoder.capped([dem(DETOUR.format(fault=0.02))])[0] == 0.1
    assert decoder.capped([dem(DETOUR.format(fault=0.0035))])[0] == 0.05
    cap, counts = decoder.capped([dem(DETOUR.format(fault=0.001))])
    assert cap is None and counts == {0.1: [1], 0.05: [1], None: [0]}
    with pytest.raises(ValueError, match="every cap"):
        decoder.capped([dem("error(0.1) D0\nerror(0.01) L0")])  # an undetectable logical


def test_a_faults_parts_keep_its_observable():
    # As stim splits a flag's fault at d=3: only boundary edges join its detectors, and the
    # part left holds the observable alone, which matching would drop.
    model = dem(
        """
        error(0.05) D0
        error(0.1) D1 L0
        error(0.02) D2
        error(0.02) D3
        error(0.002) D0 ^ D1 L0 ^ D2 ^ D3 ^ L0
        """
    )
    fired, flips = decoder.faults(model)
    assert fired[-1].tolist() == [True] * 4 and not flips[-1].any()
    for line in decoder.resplit(model).flattened():
        parts = str(line).split(")", 1)[1].split("^")
        assert all("D" in part for part in parts)  # every part has a detector
    assert not len(decoder.miscorrected(model, decoder.matching(model, None)))
    plain = pymatching.Matching.from_detector_error_model(model)
    assert decoder.decode(plain, fired[-1:])[0, 0]  # stim's split alone flips the logical


def test_graph_distance_counts_edges_or_weight():
    circuit = stim.Circuit.generated(
        "repetition_code:memory", distance=5, rounds=3, after_clifford_depolarization=0.01
    )
    model = circuit.detector_error_model(decompose_errors=True)
    assert decoder.graph_distance(decoder.matching(model, None)) == 5
    weighted = decoder.graph_distance(decoder.matching(model, None), weighted=True)
    assert weighted == pytest.approx(5 * log(99))
    assert decoder.circuit_distance(circuit) == (5, 5)


def test_single_faults_of_a_compiled_circuit():
    """The d=3 Z memory as compiled and modelled from 13 October, 2 rounds."""
    pytest.importorskip("samplomatic")
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    from followup import model, native
    from followup.circuits import Setting
    from followup.placement import place
    from heavyhex.experiment.circuits import with_durations
    from heavyhex.patches.placement import Calibration

    calibration = Calibration.load(OCTOBER_13)
    assert calibration.durations_ns is not None
    target = with_durations(FakeKingston().target, calibration.durations_ns)
    chip = place(calibration)[3].qubits
    t = native.template(Setting("memory", "Z", 3, 2), chip, target)
    noisy, _ = model.memory(native.instance(t), t.schedule, calibration, target.dt)
    errors = noisy.detector_error_model(decompose_errors=True)
    assert any(line.type == "error" and str(line).endswith("^ L0") for line in errors.flattened())
    assert len(decoder.miscorrected(errors, decoder.matching(errors, 0.1))) == 1
    assert not len(decoder.miscorrected(errors, decoder.matching(errors, None)))
    assert decoder.graph_distance(decoder.matching(errors, None)) == 3
    assert decoder.circuit_distance(noisy) == (3, 3)
