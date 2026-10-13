"""life-1: the idle test read in Z, T1 and T2 on the footprint, their fits, both idle tests
fitted together, the job read back through the analysis, and the model run again at the
data's in-situ lifetimes. Nothing here contacts IBM."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("qiskit_ibm_runtime")
pytest.importorskip("samplomatic")
pytest.importorskip("scipy")
pytest.importorskip("stim")

import numpy as np
from qiskit_ibm_runtime.fake_provider import FakeKingston
from qiskit_ibm_runtime.results.quantum_program import (
    ItemMetadata,
    QuantumProgramItemResult,
    QuantumProgramResult,
    SchedulerTiming,
)

from followup import analysis, lifetimes, model, submit
from followup.circuits import IDLE_ROUNDS, ROUNDS, Setting, logical_circuit, read_into, to_stim
from followup.placement import place
from heavyhex.experiment.circuits import with_durations
from heavyhex.patches.placement import Calibration

OCTOBER_13 = Calibration.load(
    Path(__file__).parent / "data" / "kingston_calibration_2026-10-13.json"
)
PLACEMENTS = place(OCTOBER_13)
FOOTPRINT = sorted(PLACEMENTS[5].qubits)
DATA = PLACEMENTS[5].roles["data"]


@pytest.fixture(scope="module")
def target():
    assert OCTOBER_13.durations_ns
    return with_durations(FakeKingston().target, OCTOBER_13.durations_ns)


# Circuits


@pytest.mark.parametrize("start", "01")
def test_the_idle_test_read_in_z_changes_only_the_datas_first_and_last_h(start):
    x_test, _ = logical_circuit(Setting("idle", "X", 5, 2))
    z_test, _ = logical_circuit(Setting("idle", "X", 5, 2, "Z", start))
    data = set(read_into(x_test, "d").values())

    def wires(circuit):
        found: dict[int, list[str]] = {}
        for item in circuit.data:
            for q in (circuit.find_bit(b).index for b in item.qubits):
                found.setdefault(q, []).append(item.operation.name)
        return found

    x_wires, z_wires = wires(x_test), wires(z_test)
    assert {q: w for q, w in x_wires.items() if q not in data} == {
        q: w for q, w in z_wires.items() if q not in data
    }
    for q in data:
        gates = [name for name in x_wires[q] if name != "barrier"]
        assert gates[0] == gates[-2] == "h" and gates[-1] == "measure"
        expected = (["x"] if start == "1" else []) + gates[1:-2] + ["measure"]
        assert [name for name in z_wires[q] if name != "barrier"] == expected
    model_, records = to_stim(z_test)
    read = [k for (name, _), k in records.items() if name.startswith("d")]
    assert (model_.compile_sampler(seed=0).sample(16)[:, read] == (start == "1")).all()


def test_the_idle_test_read_in_z_goes_as_exp1s_idle_test(target):
    """Twirled, its template is the X test's wire for wire, data included: the twirl's layers
    carry the data's first and last pulses, so only the angles its samplex draws differ."""
    chip = PLACEMENTS[5].qubits
    x_test = submit.twirled(Setting("idle", "X", 5, 1), chip, target)
    for start in "01":
        z_test = submit.twirled(Setting("idle", "X", 5, 1, "Z", start), chip, target, 4)
        assert (z_test.randomizations, z_test.executions) == (4, 1024)
        assert submit._wires(z_test.circuit) == submit._wires(x_test.circuit)
        assert z_test.seconds == x_test.seconds
        submit.check(z_test, randomizations=1)  # acts as its logical circuit; data read start
    (piece, *more) = submit.untwirled(Setting("idle", "X", 5, 1, "Z", "1"), chip, target, 4, 1)
    assert len(more) == 3 and piece.shots == 256 and piece.samplex is None
    submit.check(piece)


def test_t1_and_t2_wait_as_labelled_read_1_without_noise_and_get_no_decoupling(target):
    found = lifetimes.circuits(FOOTPRINT, target)
    assert [label for label, _ in found] == [
        lifetimes.label(kind, t) for kind, waits in lifetimes.WAITS_US.items() for t in waits
    ]
    for label, circuit in found:
        kind = lifetimes.kind_of(label)
        assert kind is not None
        assert lifetimes.wait_us(circuit, target.dt) == pytest.approx(float(label.split()[1]))
        assert read_into(circuit, "m") == dict(enumerate(FOOTPRINT))
        pulses = Counter(
            (item.operation.name, circuit.find_bit(item.qubits[0]).index)
            for item in circuit.data
            if item.operation.name in ("x", "sx")
        )
        assert pulses == Counter(
            {(name, q): count for q in FOOTPRINT for name, count in PULSES[kind].items()}
        )
        model_, _ = to_stim(circuit)
        assert model_.compile_sampler(seed=0).sample(8).all()  # every qubit reads 1
        submit._item(label, circuit, target, 256)  # native to the device
    assert lifetimes.kind_of("R0") is None


PULSES = {"T1": {"x": 1}, "T2": {"sx": 2, "x": 1}}  # the echo's X is its only extra pulse


def test_life_1_is_the_z_test_then_t1_and_t2_each_1024_shots(target):
    items = submit.life(PLACEMENTS, target, twirl=False)
    shots = Counter()
    for item in items:
        assert item.shots == 256
        shots[item.label] += item.executions
    labels = [str(s) for s in lifetimes.Z_IDLE] + [
        lifetimes.label(kind, t) for kind, waits in lifetimes.WAITS_US.items() for t in waits
    ]
    assert list(shots) == labels and set(shots.values()) == {lifetimes.SHOTS}
    assert len(lifetimes.Z_IDLE) == 2 * len(IDLE_ROUNDS)


# Fits


def test_decay_returns_t_with_an_error_that_covers_it_and_the_excited_share():
    rng = np.random.default_rng(3)
    t = np.array(lifetimes.WAITS_US["T1"], float)
    readout, share = 0.02, 0.015
    p = readout + (1 - 2 * readout) * (0.97 * np.exp(-t / 160.0) + share)
    pulls, excited = [], []
    for _ in range(200):
        fit = lifetimes.decay(t, rng.binomial(1024, p), 1024, 223.0)
        pulls.append((fit["t_us"] - 160.0) / fit["sigma_us"])
        pi_, sigma = lifetimes.excited(fit["offset"], fit["sigma_offset"], readout)
        excited.append((pi_ - share) / sigma)
    for found in (pulls, excited):
        assert abs(np.mean(found)) < 0.25
        assert 0.75 < np.std(found) < 1.3
    exact = lifetimes.decay(t, p * 1e9, 1e9, 50.0)
    assert exact["t_us"] == pytest.approx(160.0, rel=1e-5)
    assert lifetimes.excited(exact["offset"], 0, readout)[0] == pytest.approx(share, abs=1e-6)


def test_t_beyond_the_longest_wait_is_judged_in_its_rate():
    """At T2 twice the longest wait, T's error is too small on the low side too often; the
    rate's holds."""
    rng = np.random.default_rng(4)
    t = np.array(lifetimes.WAITS_US["T2"], float)
    true = 2 * t[-1]
    p = 0.97 * (1 + np.exp(-t / true)) / 2 + 0.015
    pulls = []
    for _ in range(200):
        fit = lifetimes.decay(t, rng.binomial(1024, p), 1024, 121.0)
        g, sigma = lifetimes.rate(fit["t_us"], fit["sigma_us"])
        pulls.append((g - 1 / true) / sigma)
    assert np.mean(np.abs(pulls) > 2) < 0.05


def chain(rounds, down, up, readout, start_ones=(0.01, 0.98)):
    """P(1) read after n rounds from |0> and |1> of a qubit flipping |1> -> |0> with `down`
    and |0> -> |1> with `up` a round, read with a symmetric error."""
    n = np.asarray(rounds, float)
    lam, share = 1 - down - up, up / (down + up)
    return [readout + (1 - 2 * readout) * (share + (one - share) * lam**n) for one in start_ones]


def test_flips_return_the_flips_a_round_of_a_two_state_chain_and_where_it_settles():
    rounds, readout = list(IDLE_ROUNDS), 0.02
    p0, p1 = chain(rounds, 0.03, 0.01, readout)
    big = np.full(len(rounds), 1e10)
    fit = lifetimes.flips(rounds, p0, p1, big, big, readout)
    assert fit["flips"][0] == pytest.approx(0.04, rel=1e-4)
    assert fit["settled"][0] == pytest.approx(0.25, rel=1e-3)
    rng = np.random.default_rng(5)
    shots = np.full(len(rounds), 1024)
    pulls = []
    for _ in range(150):
        k0, k1 = rng.binomial(1024, p0), rng.binomial(1024, p1)
        fit = lifetimes.flips(rounds, k0 / 1024, k1 / 1024, shots, shots, readout)
        pulls.append((fit["flips"][0] - 0.04) / fit["flips"][1])
    assert abs(np.mean(pulls)) < 0.3
    assert 0.7 < np.std(pulls) < 1.4


def idle_rows(t1, t_q, shots=1e12, rng=None):
    """Rows of idle() and z_idle() for one data qubit: per round 2.9 us on the equator and
    2.3 us along Z read in X, the other way round read in Z, as on the d=5 footprint."""
    n = np.array(IDLE_ROUNDS, float)
    on, off = 0.6 + 2.9 * n, 0.4 + 2.3 * n
    g = 0.93 - 0.002 * n
    r = 0.97 * g * np.exp(-on / t_q - off / t1)
    c = 0.95 * g * np.exp(-on / t1 - off / t_q)
    p0, p1 = (1 - c) / 2, (1 + c) / 2
    counts = np.full(len(n), shots)
    if rng is not None:
        r = 1 - 2 * rng.binomial(int(shots), (1 - r) / 2) / shots
        p0, p1 = (rng.binomial(int(shots), p) / shots for p in (p0, p1))
    xy = {"x": r, "y": 0 * r, "shots_x": counts, "shots_y": counts,
          "pulses": g, "equator_us": on, "along_z_us": off}  # fmt: skip
    z = {"p0": p0, "p1": p1, "shots_0": counts, "shots_1": counts,
         "pulses": g, "equator_us": off, "along_z_us": on}  # fmt: skip
    return xy, z


def test_both_idle_tests_together_give_t1_and_t_q():
    fit = analysis.together(*idle_rows(150.0, 80.0))
    assert fit["rate_t1"][0] == pytest.approx(1 / 150.0, rel=1e-4)
    assert fit["rate_t_q"][0] == pytest.approx(1 / 80.0, rel=1e-4)
    # The tests see nearly the same sum of the rates, so T1 and T_q apart are loose.
    assert fit["correlation"] < -0.9
    rng = np.random.default_rng(11)
    for _ in range(40):  # at T_q = 2 T1, the model's limit, no fit goes past it
        noisy = analysis.together(*idle_rows(150.0, 300.0, shots=2048, rng=rng))
        assert noisy["rate_t_q"][0] >= noisy["rate_t1"][0] / 2 * (1 - 1e-9)
    xy, z = idle_rows(150.0, 80.0)
    z["p1"] = z["p0"] + 0.01  # read in Z, nothing above MIN_R
    assert np.isnan(analysis.together(xy, z)["rate_t1"][0])


def test_prepared_reads_the_shortest_waits_and_one_round():
    items = [
        {"label": "T1, 20 us", "setting": None},
        {"label": "T1, 450 us", "setting": None},
        {
            "label": "z",
            "setting": {"kind": "idle", "readout": "Z", "rounds": 1, "start": "1", "distance": 5},
        },
    ]
    data = {
        "T1, 20 us": {"m": np.array([[1, 1], [1, 0]], bool)},
        "T1, 450 us": {"m": np.zeros((2, 2), bool)},
        "z": {"d5": np.array([[1, 1, 1, 0]], bool), "m5": np.zeros((1, 3), bool)},
    }
    assert lifetimes.prepared(items, data) == {"T1, 20 us": 0.75, "z": 0.75}


def test_the_table_has_each_qubit_three_ways_its_ranges_from_the_rate():
    nan = float("nan")
    situ = {
        "flips": [0.052, 0.003],
        "up": [0.0006, 0.0002],
        "rate_t1": [1 / 170, 0.002],
        "rate_t_q": [1 / 120, 0.003],
        "decay_z": [0.041, 0.002],
        "decay_z_ibm": 0.037,
        "decay_x": [0.044, 0.002],
        "decay_x_ibm": 0.040,
    }
    rows = [
        {"qubit": 6, "role": "data", "ibm": {"t1_us": 208.0, "t2_us": 100.0},
         "ours": {"rate_t1": lifetimes.rate(190.0, 15.0), "rate_t2": lifetimes.rate(95.0, 9.0),
                  "excited": [0.012, 0.004]},
         "in_situ": situ},
        {"qubit": 7, "role": "flag", "ibm": {"t1_us": 394.0, "t2_us": 70.0},
         "ours": {"rate_t1": lifetimes.rate(400.0, 80.0), "rate_t2": [nan, nan],
                  "excited": [nan, nan]},
         "in_situ": None},
        {"qubit": 8, "role": "data", "ibm": {"t1_us": 300.0, "t2_us": 320.0},
         "ours": {"rate_t1": [1e-14, 1e-3], "rate_t2": lifetimes.rate(2584.0, 12619.0),
                  "excited": [0.02, 0.5]},
         "in_situ": {**situ, "rate_t1": [0.0, 8e-4]}},
    ]  # fmt: skip
    text = lifetimes.table(rows)
    assert "   190  176-206   -1.1" in text  # 1/(1/190 ± 15/190²), (1/208 - 1/190)/σ
    assert "  2584  439-∞   " in text and "     ∞ 1000-∞   " in text  # no decay seen
    assert "     ∞ 1250-∞   " in text  # in situ, a rate on its bound
    assert "   1.20 ± 0.40" in text and "   5.20 ± 0.30" in text
    first, situ_block, _ = text.split("\n\n")
    for block in (first, situ_block):
        assert len({len(line) for line in block.splitlines()}) == 1
    assert "T1 ours/IBM: median 1.02 over 3 qubits" in text  # 190/208, 400/394, ∞
    assert "T2 ours/IBM: median 4.51 over 2 qubits" in text  # 95/100 and 2584/320
    assert "decay Z in situ/IBM: median 1.11 over 2 qubits" in text


# The job read back


def results(items, standard, situ, t_q, dt):
    """Each item's readouts as the executor returns them, from the model: T1 and T2 at
    `standard`, the idle test at `situ` with the data decaying at t_q."""

    def sampled(circuit, calibration, shots, seed, decay_us=None):
        noisy, records = model.noisy(circuit, calibration, dt, decay_us=decay_us)
        measured = noisy.compile_sampler(seed=seed).sample(shots)
        found = {}
        for register in {name for name, _ in records}:
            bits = sorted((i, k) for (name, i), k in records.items() if name == register)
            found[register] = measured[:, [k for _, k in bits]]
        return found

    out = []
    for k, item in enumerate(items):
        if item.samplex is None:
            data = sampled(item.circuit, standard, item.shots, k)
        else:
            drawn = submit.instances(item, item.randomizations, seed=k)
            parts = [sampled(c, situ, item.shots, 100 * k + i, t_q) for i, c in enumerate(drawn)]
            data = {name: np.stack([p[name] for p in parts]) for name in parts[0]}
        timing = SchedulerTiming("", round(item.seconds / dt))
        out.append(QuantumProgramItemResult(data, ItemMetadata(scheduler_timing=timing)))
    return QuantumProgramResult(out)


def saved(folder, job, target, readouts):
    """The job's record as submit.submit writes it, with these readouts."""
    folder.mkdir(parents=True)
    submit._write(folder / "run.json", submit.describe(job, PLACEMENTS, target, OCTOBER_13))
    OCTOBER_13.save(folder / "calibration.json")
    submit.save_circuits(folder, job)
    submit.save_results(folder, readouts)
    return folder


@pytest.fixture(scope="module")
def planted():
    """Lifetimes for the model to run at: T1 and T2 near IBM's, and the data in situ at 0.7
    of IBM's T1 and IBM's T2."""
    rng = np.random.default_rng(2)
    t1 = {q: OCTOBER_13.t1_us[q] * rng.uniform(0.8, 1.2) for q in FOOTPRINT}
    t2 = {q: min(OCTOBER_13.t2_us[q] * rng.uniform(0.8, 1.2), 1.9 * t1[q]) for q in FOOTPRINT}
    standard = replace(
        OCTOBER_13, t1_us={**OCTOBER_13.t1_us, **t1}, t2_us={**OCTOBER_13.t2_us, **t2}
    )
    in_place = {q: 0.7 * OCTOBER_13.t1_us[q] for q in DATA}
    t_q = {q: OCTOBER_13.t2_us[q] for q in DATA}  # as the test read in Z alone assumes
    return SimpleNamespace(
        t1=t1, t2=t2, standard=standard, in_place=in_place, t_q=t_q,
        situ=model.with_t1(OCTOBER_13, in_place),
    )  # fmt: skip


@pytest.fixture(scope="module")
def life_1(tmp_path_factory, target, planted):
    """A smaller life-1 from the model at the planted lifetimes, saved as submit saves a job."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lifetimes, "WAITS_US", {"T1": (20, 75, 245, 450), "T2": (12, 40, 132, 240)})
        patch.setattr(
            lifetimes, "Z_IDLE", tuple(s for s in lifetimes.Z_IDLE if s.rounds in (1, 16))
        )
        job = submit.Job("life-1", submit.life(PLACEMENTS, target))
    p = planted
    readouts = results(job.items, p.standard, p.situ, p.t_q, target.dt)
    return saved(tmp_path_factory.mktemp("runs") / "life-1", job, target, readouts)


@pytest.fixture(scope="module")
def exp1_1(tmp_path_factory, target, planted):
    """A smaller exp1-1 at the same lifetimes: Z memory at d=3 and the idle test at 1, 4 and
    16 rounds, 2 twirls x 256 shots each."""
    items = [
        submit.twirled(setting, PLACEMENTS[setting.distance].qubits, target, 2)
        for setting in [Setting("memory", "Z", 3, n) for n in ROUNDS]
        + [Setting("idle", "X", 5, n, readout) for readout in "XY" for n in (1, 4, 16)]
    ]
    job = submit.Job("exp1-1", tuple(items))
    p = planted
    readouts = results(job.items, p.standard, p.situ, p.t_q, target.dt)
    return saved(tmp_path_factory.mktemp("runs") / "exp1-1", job, target, readouts)


def test_life_1_read_back_through_the_analysis(life_1, planted):
    """It passes the checks, and the fits return what was planted."""
    checks = submit.inspect(life_1)
    assert checks["passed"], checks["failed"]
    assert set(checks["prepared"]) == {"T1, 20 us", "T2, 12 us"} | {
        str(s) for s in lifetimes.Z_IDLE if s.rounds == 1
    }

    found = analysis.life(life_1)
    assert [row["qubit"] for row in found["qubits"]] == FOOTPRINT
    for kind, true in (("T1", planted.t1), ("T2", planted.t2)):
        pulls = []
        for q, fit in found["standard"][kind].items():
            rate, sigma = lifetimes.rate(fit["t_us"], fit["sigma_us"])
            pulls.append((rate - 1 / true[q]) / sigma)
        assert np.median(np.abs(pulls)) < 1.2 and max(np.abs(pulls)) < 5
    rows = {row["qubit"]: row for row in found["qubits"]}
    # The model's T1, Pauli-twirled, leaves a qubit in |1> half the time at rest.
    assert 0.4 < np.median([row["ours"]["excited"][0] for row in rows.values()]) < 0.6
    z_rows = found["z_idle"]["qubits"]
    assert sorted(row["qubit"] for row in z_rows) == sorted(DATA)
    settled = [row["per_round"]["settled"][0] for row in z_rows]
    assert 0.4 < np.median(settled) < 0.6  # the twirl and XY4, whatever the qubit's own split
    pulls = [(row["t1_us"] - planted.in_place[row["qubit"]]) / row["sigma_us"] for row in z_rows]
    assert np.median(np.abs(pulls)) < 1.2 and max(np.abs(pulls)) < 5
    for q in DATA:
        situ = rows[q]["in_situ"]
        assert rows[q]["role"] == "data" and set(situ) == {"flips", "up", "rate_t1"}
        assert situ["up"][0] == pytest.approx(situ["flips"][0] * rows[q]["ours"]["excited"][0])
    assert "T1 ours/IBM" in lifetimes.table(found["qubits"])


def test_both_idle_tests_and_the_model_at_the_data_in_situ(life_1, exp1_1, planted, monkeypatch):
    """exp1(..., life=) fits both idle tests together (in_situ), which give the decay a round
    each memory's data see, as planted, and runs the model again at the data's in-situ T1 and
    T_q alone."""
    calls = []

    def predict(circuits, calibration, dt, *, decay_us, shots, seed, decoders):
        calls.append((calibration, decay_us))
        p = {key: (1 - 0.9 * 0.9 ** key[2]) / 2 for key in circuits}
        return {"rule": None, "detection": {key: {} for key in circuits},
                "decoders": {str(d): {"logical_error": p} for d in decoders}}  # fmt: skip

    monkeypatch.setattr(analysis.model, "predict", predict)
    found = analysis.exp1([exp1_1], caps=(None,), life=life_1)
    again = found["in_situ"]
    assert again["folder"] == life_1.name and len(calls) == 2
    (ibm, _), (calibration, decay_us) = calls
    assert decay_us == again["t_q_us"] and sorted(decay_us) == sorted(DATA)
    for q in FOOTPRINT:
        expected = again["t1_us"][q] if q in DATA else ibm.t1_us[q]
        assert calibration.t1_us[q] == expected
    assert set(again["fits"]) == set(again["observed_over_model"]) == {"0.1", "None"}
    o_m = again["observed_over_model"]["0.1"]["Z"]["log"]["eps3"]
    assert np.isfinite(o_m).all()

    job = analysis.record([life_1])
    read_z = analysis.z_idle(job.found, job.shots, job.calibration, job.dt)
    z_rows = {row["qubit"]: row for row in read_z["qubits"]}
    pulls = []
    for row in again["qubits"]:
        q = row["qubit"]
        times = analysis._per_round(z_rows[q], read_z["rounds"])
        t1 = planted.in_place[q]
        true = times @ [1 / t1, 1 / min(planted.t_q[q], 2 * t1)]  # the model's cap
        pulls.append((row["decay_z"][0] - true) / row["decay_z"][1])
        assert row["correlation"] < -0.5 and row["decay_z_ibm"] < true
    assert np.median(np.abs(pulls)) < 1.5 and max(np.abs(pulls)) < 5


def test_axis_times_give_the_data_round_by_round_on_both_axes(target):
    """Read in Z, a d=5 round holds a data qubit about 2.9 us along Z and 2.3 us on the
    equator; read in X, the other way round."""
    chip = PLACEMENTS[5].qubits
    per_round = {}
    for readout, start in (("X", "+"), ("Z", "1")):
        times = []
        for n in (1, 2):
            item = submit.twirled(Setting("idle", "X", 5, n, readout, start), chip, target)
            (bound,) = submit.instances(item, 1)
            found = analysis.axis_times(bound, DATA, target.dt)
            times.append({side: np.array([found[side][q] for q in DATA]) for side in found})
        per_round[readout] = {side: times[1][side] - times[0][side] for side in times[0]}
    assert np.allclose(per_round["X"]["equator"], per_round["Z"]["Z"])
    assert np.allclose(per_round["X"]["Z"], per_round["Z"]["equator"])
    assert 2.5 < np.median(per_round["Z"]["Z"]) < 3.3
    assert 1.9 < np.median(per_round["Z"]["equator"]) < 2.7
