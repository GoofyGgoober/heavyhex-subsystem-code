"""The hardware run: native circuits, their timing, and the analysis, all local."""

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

pytest.importorskip("stim")
pytest.importorskip("pymatching")
pytest.importorskip("qiskit_aer")
pytest.importorskip("qiskit_ibm_runtime")

import numpy as np  # noqa: E402
from qiskit_ibm_runtime.fake_provider import FakeFez  # noqa: E402

import heavyhex.simulation.noisy as noisy  # noqa: E402
from heavyhex.cli import main  # noqa: E402
from heavyhex.experiment import run  # noqa: E402
from heavyhex.experiment.circuits import Setting, for_fez, logical_circuit  # noqa: E402
from heavyhex.patches.placement import LAST_CALIBRATION, Calibration, fez_qubits  # noqa: E402

TARGET = FakeFez().target
SAVED = Calibration.load(LAST_CALIBRATION)
WORKING = replace(SAVED, cz=dict.fromkeys(SAVED.cz, 0.003))  # so both patches can be placed
pytestmark = pytest.mark.filterwarnings("ignore:.*weak parts")


def chips():
    return {d: fez_qubits(d, WORKING) for d in (3, 5)}


def native(setting):
    return for_fez(logical_circuit(setting, chips(), TARGET.num_qubits)[0], TARGET)


@pytest.mark.parametrize("setting", [Setting("memory", "X", 2), Setting("idle", "X", 2)])
def test_native_circuits_use_only_the_chips_instructions(setting):
    circuit = native(setting)
    for item in circuit.data:
        qubits = tuple(circuit.find_bit(q).index for q in item.qubits)
        if item.operation.name != "barrier":
            assert TARGET.instruction_supported(item.operation.name, qubits)
        if item.operation.name == "delay":
            assert item.operation.duration >= TARGET.min_length


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_pulses_land_where_the_simulator_puts_them(basis):
    circuit = native(Setting("memory", basis, 2))
    sent = Counter(circuit.find_bit(i.qubits[0]).index for i in circuit.data if i.name == "x")
    for d, chip in chips().items():
        model = noisy.noisy_circuit(d, basis, 2, WORKING, decoupling=True)
        simulated = Counter(
            chip[t.value] for i in model.flattened() if i.name == "X" for t in i.targets_copy()
        )
        assert {q: sent[q] for q in chip} == {q: simulated[q] for q in chip}


def test_a_noiseless_chip_never_fires_a_detector():
    from qiskit_aer import AerSimulator
    from qiskit_ibm_runtime import SamplerV2

    settings = [Setting("memory", "X", 2), Setting("memory", "Z", 2), Setting("idle", "X", 2)]
    circuits = [native(s) for s in settings]
    result = SamplerV2(mode=AerSimulator(method="stabilizer")).run(circuits, shots=50).result()
    for setting, circuit, pub in zip(settings, circuits, result):
        bits = {name: b.to_bool_array(order="little") for name, b in pub.data.items()}
        model, records, _ = run.simulated(setting, circuit, WORKING, TARGET.dt, 0.0)
        measured = run.in_record_order(bits, records)
        if setting.kind == "memory":
            detectors, flips = model.compile_m2d_converter().convert(
                measurements=measured, separate_observables=True
            )
            assert not detectors.any() and not flips.any()
        else:
            assert all(x == 1 for xs in run.x_values(bits).values() for x in xs)


def test_shots_go_to_registers_and_back():
    circuit = native(Setting("memory", "Z", 1))
    model, records, _ = run.simulated(Setting("memory", "Z", 1), circuit, WORKING, TARGET.dt, 0)
    measured = model.compile_sampler(seed=3).sample(20)
    assert (run.in_record_order(run.by_register(measured, records), records) == measured).all()


def test_a_rehearsal_gives_back_its_dephasing(tmp_path, monkeypatch):
    settings = (
        (Setting("memory", "X", 1), Setting("memory", "X", 2))
        + (
            Setting("memory", "Z", 1),
            Setting("memory", "Z", 2),
        )
        + tuple(Setting("idle", "X", n) for n in (1, 4, 8))
    )
    monkeypatch.setattr(run, "SETTINGS", settings)
    monkeypatch.setattr(run, "ROUNDS", (1, 2))
    monkeypatch.setattr(run, "IDLE_ROUNDS", (1, 4, 8))
    monkeypatch.setattr(run, "SIMULATED_SHOTS", 20_000)
    run.prepare(tmp_path / "run", WORKING, TARGET, offline=True)
    run.rehearse(tmp_path / "run", 20_000, decoupled_dephasing=0.5)
    analysis = run.analyze(tmp_path / "run")
    assert analysis["decoupled_dephasing"] == pytest.approx(0.5, abs=0.1)
    assert 0 < analysis["observed"]["fits"]["X"]["lambda_uncertainty"] < 0.2
    seen = analysis["observed"]["settings"]
    expected = analysis["predicted_at_measured_dephasing"]["settings"]
    for got, want in zip(seen[:4], expected[:4]):
        assert np.allclose(got["logical_error"], want["logical_error"], atol=0.015)


def test_submit_refuses_a_run_prepared_offline(tmp_path, capsys):
    (tmp_path / "run.json").write_text(json.dumps({"offline": True, "prepared": "x"}))
    assert main(["experiment", "submit", "--run", str(tmp_path)]) == 2
    assert "prepared offline" in capsys.readouterr().err


def test_submit_refuses_yesterdays_run(tmp_path, capsys):
    prepared = "2020-01-01T10:00:00+00:00"
    (tmp_path / "run.json").write_text(json.dumps({"offline": False, "prepared": prepared}))
    assert main(["experiment", "submit", "--run", str(tmp_path)]) == 2
    assert "prepare it again" in capsys.readouterr().err


def test_default_folder_is_under_runs():
    assert run.default_folder("ibm_fez", offline=False).parent == Path(run.RUNS)


def test_the_idle_test_waits_and_pulses_like_x_memory():
    def pulses_and_length(setting):
        circuit = native(setting)
        data = [q for d, chip in chips().items() for q in chip[: d * d]]
        sent = Counter(circuit.find_bit(i.qubits[0]).index for i in circuit.data if i.name == "x")
        return [sent[q] for q in data], circuit.estimate_duration(TARGET, unit="s")

    idle, memory = (
        pulses_and_length(Setting("idle", "X", 2)),
        pulses_and_length(Setting("memory", "X", 2)),
    )
    assert idle[0] == memory[0]
    assert idle[1] == pytest.approx(memory[1])
