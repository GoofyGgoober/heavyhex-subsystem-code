"""The follow-up's circuits in ibm_kingston's native gates: twirled boxes, ALAP and XY4."""

from pathlib import Path

import pytest

pytest.importorskip("qiskit_ibm_runtime")
pytest.importorskip("samplomatic")
pytest.importorskip("stim")

from qiskit_ibm_runtime.fake_provider import FakeKingston

from followup import native
from followup.circuits import Schedule, Setting, add_detectors, noiseless_check, to_stim
from followup.placement import place
from heavyhex.circuits.flagged import FlaggedSchedule
from heavyhex.experiment.circuits import with_durations
from heavyhex.patches.placement import Calibration
from heavyhex.simulation.noisy import add_detectors as run1_detectors

OCTOBER_13 = Calibration.load(
    Path(__file__).parent / "data" / "kingston_calibration_2026-10-13.json"
)
XY4_LENGTH = 4 * 8  # four x pulses of 32 ns, in 4 ns steps


@pytest.fixture(scope="module")
def target():
    assert OCTOBER_13.durations_ns is not None
    return with_durations(FakeKingston().target, OCTOBER_13.durations_ns)


@pytest.fixture(scope="module")
def chips():
    return {d: p.qubits for d, p in place(OCTOBER_13).items()}


def stim_memory(circuit, schedule):
    out, records = to_stim(circuit)
    add_detectors(out, schedule, {(name[0], i): k for (name, i), k in records.items()})
    return out


def by_qubit(circuit):
    """Each qubit's instructions, in order, as (name, duration)."""
    found = {}
    for item in circuit.data:
        for q in item.qubits:
            op = item.operation
            length = op.duration if op.name == "delay" else None
            found.setdefault(circuit.find_bit(q).index, []).append((op.name, length))
    return found


@pytest.mark.parametrize("basis", "XZ")
@pytest.mark.parametrize("distance", [3, 5])
def test_every_randomization_keeps_the_detectors(target, chips, distance, basis):
    t = native.template(Setting("memory", basis, distance, 2), chips[distance], target)
    assert t.samplex is not None and t.circuit.num_parameters
    for seed in (0, 1):
        noiseless_check(stim_memory(native.instance(t, seed), t.schedule))


def test_twirling_gives_each_layers_qubits_two_sx(target, chips):
    twirled = native.template(Setting("memory", "X", 3, 2), chips[3], target)
    plain = native.template(Setting("memory", "X", 3, 2), chips[3], target, twirl=False)
    sx = [
        {q: sum(name == "sx" for name, _ in steps) for q, steps in by_qubit(t.circuit).items()}
        for t in (twirled, plain)
    ]
    assert all(sx[0][q] > sx[1][q] for q in chips[3])
    assert sum(sx[0].values()) > 1.5 * sum(sx[1].values())
    # Most layers take the place of the H gates around a CZ, so the circuit is little longer.
    lengths = [t.circuit.estimate_duration(target) for t in (twirled, plain)]
    assert lengths[1] < lengths[0] < 1.1 * lengths[1]


def test_xy4_fills_every_idle_it_fits(target, chips):
    circuit = native.template(Setting("memory", "Z", 3, 2), chips[3], target).circuit
    pulses = 0
    for q in chips[3]:
        steps = [step for step in by_qubit(circuit)[q] if step[0] not in ("rz", "barrier")]
        pulses += sum(name == "x" for name, _ in steps)
        for k, (name, length) in enumerate(steps):
            near = {steps[j][0] for j in (k - 1, k + 1) if 0 <= j < len(steps)}
            if name == "delay" and k and "x" not in near:
                assert length < XY4_LENGTH, (q, k)
    assert pulses and pulses % 4 == 0


def test_the_rounds_without_resets_are_shorter(target, chips):
    def round_us(circuits, basis, d):
        seconds = [circuits[basis, d, n][0].estimate_duration(target, unit="s") for n in (2, 4)]
        return (seconds[1] - seconds[0]) / 2 * 1e6

    new, old = native.memory_circuits(chips, target), native.run1_circuits(chips, target)
    for basis in "XZ":
        for d in (3, 5):
            assert 5.7 < round_us(new, basis, d) < 5.9
            assert 10.1 < round_us(old, basis, d) < 10.3


@pytest.mark.parametrize("readout", "XY")
def test_the_idle_test_waits_as_memory_does(target, chips, readout):
    idle = native.template(Setting("idle", "X", 5, 2, readout), chips[5], target)
    memory = native.template(Setting("memory", "X", 5, 2), chips[5], target)
    circuit = native.instance(idle)
    assert "cz" not in circuit.count_ops()
    assert idle.circuit.estimate_duration(target) == memory.circuit.estimate_duration(target)
    steps = [by_qubit(t.circuit) for t in (idle, memory)]
    for q in chips[5]:
        assert [name for name, _ in steps[0][q] if name != "rz"] == [
            "delay" if name == "cz" else name for name, _ in steps[1][q] if name != "rz"
        ]
    out, records = to_stim(circuit)
    data = sorted(k for (name, _), k in records.items() if name == "d5")
    shots = out.compile_sampler(seed=1).sample(200)[:, data]
    if readout == "X":
        assert not shots.any()  # nothing acts on the data: |+> stays |+>
    else:
        assert 0.3 < shots.mean() < 0.7


def test_run1s_resets_can_come_back(target, chips):
    t = native.template(Setting("memory", "Z", 3, 2), chips[3], target, resets=True)
    assert isinstance(t.schedule, FlaggedSchedule) and not isinstance(t.schedule, Schedule)
    circuit = native.instance(t)
    assert circuit.count_ops()["reset"] > 0
    out, records = to_stim(circuit)
    run1_detectors(out, t.schedule, {(name[0], i): k for (name, i), k in records.items()})
    noiseless_check(out)
