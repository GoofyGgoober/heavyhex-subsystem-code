"""The hardware run: native circuits, their timing, and the analysis, all local."""

import json
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

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
# Every CZ working, so both patches can be placed, and FakeFez's gate times, as TARGET has.
WORKING = replace(SAVED, cz=dict.fromkeys(SAVED.cz, 0.003), durations_ns=None)
pytestmark = pytest.mark.filterwarnings("ignore:.*weak parts")


@pytest.fixture(autouse=True)
def scratch_runs(tmp_path, monkeypatch):
    """Run folders go to a scratch place, so the repository's own runs and jobs never count."""
    monkeypatch.setattr(run, "RUNS", tmp_path / "runs")


def chips():
    return {"3": fez_qubits(3, WORKING), "5": fez_qubits(5, WORKING)}


def data_qubits(patches=("3", "5")):
    return [q for p in patches for q in chips()[p][: int(p[0]) ** 2]]


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
    for d, chip in ((3, chips()["3"]), (5, chips()["5"])):
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
        + tuple(Setting("idle", "X", n, readout=r) for r in "XY" for n in (1, 4, 8))
    )
    monkeypatch.setattr(run, "SETTINGS", settings)
    monkeypatch.setattr(run, "ROUNDS", (1, 2))
    monkeypatch.setattr(run, "SIMULATED_SHOTS", 20_000)
    monkeypatch.setattr(run, "SIDE_SHOTS", 5_000)
    folder = tmp_path / "run"
    run.prepare(folder, WORKING, TARGET, offline=True)
    shutil.copy(folder / "calibration.json", folder / "calibration_after.json")
    run.rehearse(folder, 20_000, decoupled_dephasing=0.5)
    assert not (folder / run.SHOTS).exists()  # the chip's shots are never faked
    analysis = run.analyze(folder, rehearsal=True)
    assert analysis["shots_from"] == run.REHEARSAL
    assert analysis["decoupled_dephasing"] == pytest.approx(0.5, abs=0.1)
    found = {row["qubit"]: row for row in analysis["dephasing_by_qubit"]}
    assert len(found) == 9 + 25
    measured = [row["dephasing"] for row in found.values() if row["measured"]]
    assert len(measured) > 25 and np.median(measured) == pytest.approx(0.5, abs=0.1)
    # The simulator has no frequency offsets, so <Y> stays at 0 and the phase with it.
    phases = [row["phase_per_round"] for row in found.values() if row["measured"]]
    phases = [phase for phase in phases if np.isfinite(phase)]  # NaN: too few lengths to tell
    assert len(phases) > 25 and all(abs(phase) < 0.05 for phase in phases)
    assert analysis["idle_test"]["x_only_median"] == pytest.approx(0.5, abs=0.1)
    assert 0 < analysis["observed"]["fits"]["X"]["lambda_uncertainty"] < 0.2
    seen = analysis["observed"]["settings"]
    expected = analysis["predicted_at_measured_dephasing"]["settings"]
    for got, want in zip(seen[:4], expected[:4]):
        assert np.allclose(got["logical_error"], want["logical_error"], atol=0.015)
    # Six 2σ checks: by chance one misses about one rehearsal in four, so ask for 3σ here.
    rows = analysis["agreement"]["tests"].values()
    assert all(abs(row["observed"] - row["predicted"]) < 1.5 * row["bound"] for row in rows)
    assert set(analysis["agreement"]["matches"]) == {"X", "Z"}
    assert not any(row["calibration_gap"] for row in rows)  # the same calibration after
    assert set(analysis["drift"]) == {f"{b} eps{d}" for b in "XZ" for d in (3, 5)}
    memory = analysis["dephasing_from_memory"]  # two rounds only, so ε₃ pins f to about ±0.06
    assert abs(memory["from_eps3"] - 0.5) < 3 * memory["from_eps3_uncertainty"]
    assert analysis["detectors_beyond_chance"] < 10
    assert not any(row["drift"] for row in analysis["drift"].values())
    assert not any(row["rises"] for row in analysis["leakage"].values())
    assert analysis["idle_test"]["uniformity"]["qubits"] > 25


def test_a_wait_without_echo_room_is_left_to_the_median():
    from heavyhex.experiment.run import _fit_share

    seen = np.array([0.9, 0.8, 0.6])
    flat = np.log([0.9, 0.8, 0.6])  # IBM's T2 adds nothing: T2 is 2 T1
    share, sigma = _fit_share(seen, flat, flat - 1e-4, np.full(3, 2500))
    assert np.isnan(share) and np.isnan(sigma)
    n = np.array([1, 2, 4])
    log_x0, log_x1 = -0.02 * n, -0.12 * n  # IBM's T2 adds 0.1 per round
    share, sigma = _fit_share(np.exp(-0.07 * n), log_x0, log_x1, np.full(3, 2500))
    assert share == pytest.approx(0.5) and sigma < 0.5


def test_the_job_runs_every_circuit_in_both_blocks_with_equal_pieces():
    from heavyhex.experiment.circuits import BLOCKS, SETTINGS, job_order, piece_shots

    order = job_order(SETTINGS)
    for block in range(BLOCKS):
        mine = Counter(i for i, b in order if b == block)
        assert mine == {i: s.copies for i, s in enumerate(SETTINGS)}
    assert [i for i, b in order if b == 0] != [i for i, b in order if b == 1]
    each = piece_shots(5000)
    shots = Counter()
    for i, _ in order:
        shots[SETTINGS[i]] += each
    assert all(shots[s] == (5000 if s.decoded else 2500) for s in SETTINGS)
    with pytest.raises(ValueError, match="multiple of 4"):
        piece_shots(5001)


def test_the_idle_test_reads_y_after_an_s_dagger():
    stim_circuit = native(Setting("idle", "X", 1, readout="Y"))
    plain = native(Setting("idle", "X", 1))
    turns = Counter(round(i.operation.params[0], 6) for i in stim_circuit.data if i.name == "rz")
    before = Counter(round(i.operation.params[0], 6) for i in plain.data if i.name == "rz")
    assert turns - before == Counter({round(-np.pi / 2, 6): 9 + 25})
    model, records, _ = run.simulated(
        Setting("idle", "X", 1, readout="Y"), stim_circuit, WORKING, TARGET.dt, 0.0
    )
    y = run.x_values(run.by_register(model.compile_sampler(seed=1).sample(4000), records))
    assert all(abs(v) < 0.1 for vs in y.values() for v in vs)  # |+> has no Y


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_the_circuits_sent_to_the_chip_have_distance_3_and_5(basis):
    import stim

    model, _, _ = run.simulated(
        Setting("memory", basis, 2), native(Setting("memory", basis, 2)), WORKING, TARGET.dt, 0.5
    )
    for k, d in enumerate((3, 5)):
        alone = stim.Circuit()
        for item in model.flattened():
            if item.name == "OBSERVABLE_INCLUDE" and item.gate_args_copy() != [k]:
                continue
            alone.append(item)
        assert len(alone.shortest_graphlike_error(canonicalize_circuit_errors=True)) == d


def test_a_qubit_can_have_its_own_dephasing():
    from heavyhex.simulation.scheduled import noisy_scheduled

    circuit = native(Setting("idle", "X", 2))
    mine, other = chips()["3"][:2]  # two data qubits

    def dephasing(model, qubit):
        return sum(
            op.gate_args_copy()[2]
            for op in model.flattened()
            if op.name == "PAULI_CHANNEL_1" and op.targets_copy()[0].value == qubit
        )

    shared, _ = noisy_scheduled(circuit, WORKING, TARGET.dt, decoupled_dephasing=0.0)
    own, _ = noisy_scheduled(
        circuit, WORKING, TARGET.dt, decoupled_dephasing=0.0, dephasing_by_qubit={mine: 1.0}
    )
    assert dephasing(own, mine) > dephasing(shared, mine)
    assert dephasing(own, other) == dephasing(shared, other)


def test_submit_refuses_a_run_prepared_offline(tmp_path, capsys):
    (tmp_path / "run.json").write_text(json.dumps({"offline": True, "prepared": "x"}))
    assert main(["experiment", "submit", "--run", str(tmp_path)]) == 2
    assert "prepared offline" in capsys.readouterr().err


def prepared_today(folder, seconds=1e-4):
    from dataclasses import asdict

    from heavyhex.experiment.circuits import SETTINGS, job_order

    info = {
        "offline": False,
        "prepared": run._now(),
        "settings": [asdict(s) for s in SETTINGS],
        "order": job_order(SETTINGS),
        "code": CLEAN,
        "seconds": [seconds] * len(SETTINGS),  # each circuit's length
    }
    (folder / "run.json").write_text(json.dumps(info))


CLEAN = {"commit": "c0de", "code": ["a", "b"], "changed": False}


def test_submit_refuses_a_run_that_isnt_committed(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(run, "commit", lambda: CLEAN)
    prepared_today(tmp_path)
    assert main(["experiment", "submit", "--run", str(tmp_path)]) == 2
    assert "commit and push" in capsys.readouterr().err


def test_submit_refuses_a_run_that_already_has_a_job(tmp_path, capsys):
    prepared_today(tmp_path)
    (tmp_path / "job.json").write_text("{}")
    assert main(["experiment", "submit", "--run", str(tmp_path)]) == 2
    assert "already has a job" in capsys.readouterr().err


def test_a_run_prepared_with_other_settings_is_refused(tmp_path, capsys):
    prepared_today(tmp_path)
    info = json.loads((tmp_path / "run.json").read_text())
    info["settings"] = info["settings"][1:]
    (tmp_path / "run.json").write_text(json.dumps(info))
    assert main(["experiment", "submit", "--run", str(tmp_path)]) == 2
    assert "other settings" in capsys.readouterr().err


def test_a_run_folder_must_be_pushed_as_well_as_committed(tmp_path, monkeypatch):
    import subprocess

    def git(*words, cwd):
        subprocess.run(["git", *words], cwd=cwd, check=True, capture_output=True)

    origin, clone = tmp_path / "origin.git", tmp_path / "clone"
    git("init", "--bare", str(origin), cwd=tmp_path)
    git("clone", str(origin), str(clone), cwd=tmp_path)
    for key, value in (("user.name", "t"), ("user.email", "t@t"), ("commit.gpgsign", "false")):
        git("config", key, value, cwd=clone)
    folder = clone / "runs" / "x"
    folder.mkdir(parents=True)
    (folder / "run.json").write_text("{}")
    monkeypatch.setattr(run, "REPO", clone)
    assert run.uncommitted(folder)
    git("add", ".", cwd=clone)
    git("commit", "-m", "run", cwd=clone)
    assert run.uncommitted(folder) == ["(not pushed)"]
    git("push", "origin", "HEAD", cwd=clone)
    git("branch", "--set-upstream-to", f"origin/{current_branch(clone)}", cwd=clone)
    assert run.uncommitted(folder) == []


def current_branch(clone):
    import subprocess

    return subprocess.run(
        ["git", "branch", "--show-current"], cwd=clone, capture_output=True, text=True
    ).stdout.strip()


def test_submit_refuses_yesterdays_run(tmp_path, capsys):
    prepared = "2020-01-01T10:00:00+00:00"
    (tmp_path / "run.json").write_text(json.dumps({"offline": False, "prepared": prepared}))
    assert main(["experiment", "submit", "--run", str(tmp_path)]) == 2
    assert "prepare it again" in capsys.readouterr().err


QPU = SimpleNamespace(
    name="ibm_fez", configuration=lambda: SimpleNamespace(default_rep_delay=250e-6)
)


def fake_ibm(monkeypatch, tmp_path, ibm_estimate=None, reply_lost=False):
    """IBM's sampler and jobs stood in for, recording what submit asks. No QPU is involved."""
    import qiskit_ibm_runtime
    from qiskit_ibm_runtime.exceptions import IBMRuntimeError, RuntimeJobMaxTimeoutError

    from heavyhex.experiment.circuits import SETTINGS

    sent = SimpleNamespace(options=None, circuits=None, cancelled=False)

    class Job:  # as IBM leaves a job it stopped at the limit
        usage_estimation = {"quantum_seconds": ibm_estimate}

        def job_id(self):
            return "fake"

        def cancel(self):
            sent.cancelled = True

        def result(self):
            raise RuntimeJobMaxTimeoutError("Error code 1305; Ran too long")

        def metrics(self):
            return {"timestamps": {}, "usage": {"quantum_seconds": 50, "status": "complete"}}

        def status(self):
            return "ERROR"

        def error_message(self):
            return "Ran too long"

    class Sampler:
        def __init__(self, mode):
            self.options = sent.options = SimpleNamespace(
                default_shots=None,
                max_execution_time=None,
                environment=SimpleNamespace(job_tags=None),
                dynamical_decoupling=SimpleNamespace(enable=True),
                twirling=SimpleNamespace(enable_gates=True, enable_measure=True),
            )

        def run(self, circuits):
            sent.circuits = circuits
            if reply_lost:  # IBM took the job, but its reply never came
                raise IBMRuntimeError("Failed to run program: read timed out")
            return Job()

    class Service:
        def job(self, job_id):
            return Job()

    monkeypatch.setattr(qiskit_ibm_runtime, "SamplerV2", Sampler)
    monkeypatch.setattr(qiskit_ibm_runtime, "QiskitRuntimeService", Service)
    monkeypatch.setattr(run, "load_circuits", lambda folder: list(range(len(SETTINGS))))
    monkeypatch.setattr(run, "commit", lambda: CLEAN)
    folder = run.RUNS / "today"
    folder.mkdir(parents=True)
    return sent, folder


def test_ibm_cancels_the_job_past_50_s_of_qpu_time_and_it_is_never_sent_again(
    tmp_path, monkeypatch
):
    from qiskit_ibm_runtime.exceptions import RuntimeJobMaxTimeoutError

    sent, folder = fake_ibm(monkeypatch, tmp_path, ibm_estimate=30.0)
    prepared_today(folder)  # 2 s + 68 pieces x 1,250 shots x 350 us: about 32 s
    with pytest.raises(RuntimeJobMaxTimeoutError, match="Ran too long"):
        run.submit(folder, QPU, 5000)
    assert sent.options.max_execution_time == run.QPU_SECONDS_LIMIT == 50
    assert sent.options.environment.job_tags == ["heavyhex:today"]
    assert not sent.options.dynamical_decoupling.enable
    record = json.loads((folder / "job.json").read_text())
    assert record["job"] == "fake" and record["qpu_seconds_limit"] == 50
    assert record["qpu_seconds_estimate"] == pytest.approx(2 + 68 * 1250 * 350e-6)
    assert record["status"] == "ERROR" and record["reason"] == "Ran too long"
    assert record["usage"]["quantum_seconds"] == 50
    with pytest.raises(ValueError, match="already has a job"):
        run.submit(folder, QPU, 5000)
    with pytest.raises(ValueError, match="ended ERROR without shots .Ran too long."):
        run.fetch(folder)  # analyze says so, and sends nothing


def test_a_job_estimated_past_the_limit_is_not_sent(tmp_path, monkeypatch):
    sent, folder = fake_ibm(monkeypatch, tmp_path)
    prepared_today(folder, seconds=1e-3)  # about 108 s
    with pytest.raises(ValueError, match="over its 50 s limit .* not sent"):
        run.submit(folder, QPU, 5000)
    assert sent.circuits is None and not (folder / "job.json").exists()


def test_a_job_ibm_estimates_past_the_limit_is_cancelled_before_it_runs(tmp_path, monkeypatch):
    sent, folder = fake_ibm(monkeypatch, tmp_path, ibm_estimate=80.0)
    prepared_today(folder)
    with pytest.raises(ValueError, match="cancelled before it ran"):
        run.submit(folder, QPU, 5000)
    assert sent.cancelled
    assert json.loads((folder / "job.json").read_text())["qpu_seconds_ibm_estimate"] == 80.0


def test_a_job_whose_reply_from_ibm_is_lost_is_never_sent_again(tmp_path, monkeypatch):
    from qiskit_ibm_runtime.exceptions import IBMRuntimeError

    sent, folder = fake_ibm(monkeypatch, tmp_path, reply_lost=True)
    prepared_today(folder)
    with pytest.raises(IBMRuntimeError):
        run.submit(folder, QPU, 5000)
    record = json.loads((folder / "job.json").read_text())
    assert record["job"] is None and record["tag"] == "heavyhex:today"
    with pytest.raises(ValueError, match="already has a job"):
        run.submit(folder, QPU, 5000)
    with pytest.raises(ValueError, match="reply never came.*job_tags"):
        run.fetch(folder)


def earlier_job(name, **record):
    folder = run.RUNS / name
    folder.mkdir(parents=True)
    (folder / "job.json").write_text(json.dumps(record) if record else "")


def test_the_jobs_count_ibms_usage_or_their_whole_limit(monkeypatch):
    earlier_job("counted", usage={"quantum_seconds": 17, "status": "complete"})
    earlier_job("still counting", qpu_seconds_limit=40, usage={"quantum_seconds": 3})
    earlier_job("reply lost", job=None, qpu_seconds_limit=30)
    earlier_job("unreadable")  # an empty record counts as a whole job
    earlier_job("local", local=True, usage={"quantum_seconds": 99, "status": "complete"})
    assert run.qpu_committed() == 17 + 40 + 30 + run.QPU_SECONDS_LIMIT
    assert run.job_limit() == min(run.QPU_SECONDS_LIMIT, 300 - 137) == 50
    earlier_job("big", usage={"quantum_seconds": 150, "status": "complete"})
    assert run.job_limit() == 300 - 287 == 13


def test_every_job_together_stays_within_300_s_of_qpu_time(tmp_path, monkeypatch):
    sent, folder = fake_ibm(monkeypatch, tmp_path, ibm_estimate=10.0)
    prepared_today(folder)  # 2 s + 68 pieces x 625 shots x 350 us: about 17 s
    earlier_job("earlier", usage={"quantum_seconds": 270, "status": "complete"})
    with pytest.raises(Exception, match="Ran too long"):
        run.submit(folder, QPU, 2500)
    assert sent.options.max_execution_time == 30  # what the 300 s had left
    assert json.loads((folder / "job.json").read_text())["qpu_seconds_limit"] == 30
    sent.circuits = None
    other = run.RUNS / "other"
    other.mkdir()
    prepared_today(other)
    with pytest.raises(ValueError, match="jobs so far count 320 s of the 300 s"):
        run.submit(other, QPU, 2500)  # the first job counts 50 s until IBM's count is final
    assert sent.circuits is None and not (other / "job.json").exists()


def test_a_job_estimated_past_what_the_total_has_left_is_not_sent(tmp_path, monkeypatch):
    sent, folder = fake_ibm(monkeypatch, tmp_path)
    prepared_today(folder)
    earlier_job("earlier", usage={"quantum_seconds": 290, "status": "complete"})
    with pytest.raises(ValueError, match="over its 10 s limit .*290 s already counted"):
        run.submit(folder, QPU, 2500)
    assert sent.circuits is None and not (folder / "job.json").exists()


def test_no_job_goes_past_the_approved_count(tmp_path, monkeypatch):
    sent, folder = fake_ibm(monkeypatch, tmp_path)
    prepared_today(folder)
    for k in range(run.QPU_JOBS_LIMIT):
        earlier_job(f"job {k}", usage={"quantum_seconds": 0, "status": "complete"})
    with pytest.raises(ValueError, match="20 jobs have gone, and 20 were approved"):
        run.submit(folder, QPU, 2500)
    assert sent.circuits is None


def test_drift_sets_the_first_half_of_every_pieces_shots_against_the_second(tmp_path):
    import numpy as np

    ran = np.array([[0], [0], [1], [1]], dtype=bool)  # shots in the order they ran
    np.savez(tmp_path / run.SHOTS, **{"0/0/0/m": ran, "1/0/1/m": ran})
    first, second = (run.load_shots(tmp_path, half=half)[0]["m"] for half in (0, 1))
    assert not first.any() and second.all() and len(first) == len(second) == 4


def test_day_folders_are_named_by_day_rep_and_lone_patch():
    assert run.day_folder("ibm_fez", "2026-10-04", 1) == run.RUNS / "ibm_fez-2026-10-04-r1"
    assert run.day_folder("ibm_fez", "2026-10-04", 2, 5).name == "ibm_fez-2026-10-04-r2-d5"
    assert run.day_folder("ibm_fez", "2026-10-04", 2, 3, offline=True).name.endswith("d3-offline")


OCTOBER_3 = Calibration.load(Path(__file__).parent / "data" / "fez_calibration_2026-10-03.json")
SMALL = (
    (Setting("memory", "X", 1), Setting("memory", "X", 2))
    + (Setting("memory", "Z", 1), Setting("memory", "Z", 2))
    + tuple(Setting("idle", "X", n, readout=r) for r in "XY" for n in (1, 4, 8))
)


@pytest.fixture
def small_run(monkeypatch):
    """Two rounds and few shots, so a whole day runs in seconds."""
    monkeypatch.setattr(run, "SETTINGS", SMALL)
    monkeypatch.setattr(run, "ROUNDS", (1, 2))
    monkeypatch.setattr(run, "SIMULATED_SHOTS", 10_000)
    monkeypatch.setattr(run, "SIDE_SHOTS", 2_000)


def test_a_day_gets_a_folder_for_each_job_in_the_order_they_are_sent(small_run):
    prepared = run.prepare_day(OCTOBER_3, TARGET, offline=True)
    infos = [json.loads((folder / "run.json").read_text()) for folder, _ in prepared]
    day = infos[0]["rep"]["day"]
    assert day == infos[0]["prepared"][:10]
    names = [folder.name for folder, _ in prepared]
    assert names == [f"ibm_fez-{day}-r{r}-offline" for r in ("1", "2-d5", "2-d3")]
    assert [sorted(info["chips"]) for info in infos] == [["3", "5"], ["5"], ["3"]]
    assert [(i["rep"]["rep"], i["rep"]["pairing"], i["rep"]["job"]) for i in infos] == [
        (1, "paired", 1),
        (2, "split", 1),
        (2, "split", 2),
    ]
    alone = prepared[1][1]["0.0"]["fits"]["X"]
    assert set(alone["per_round"]) == {"5"} and "lambda" not in alone


def test_a_split_rep_gets_its_lambda_from_both_jobs_and_every_rep_is_pooled(small_run, capsys):
    jobs = [
        ("r1", 1, "paired", {"3": fez_qubits(3, WORKING), "5": fez_qubits(5, WORKING)}),
        ("r2-d5", 2, "split", {"5": fez_qubits(5, WORKING)}),
        ("r2-d3", 2, "split", {"3": fez_qubits(3, WORKING)}),
    ]
    for k, (name, r, pairing, chips) in enumerate(jobs):
        folder = run.RUNS / f"ibm_fez-2026-10-04-{name}"
        rep = {"day": "2026-10-04", "rep": r, "pairing": pairing, "job": 1, "spots": {}}
        run.prepare(folder, WORKING, TARGET, offline=True, chips=chips, rep=rep)
        run.rehearse(folder, 10_000, decoupled_dephasing=0.5, seed=10 * k)
        analysis = run.analyze(folder, rehearsal=True)
        assert analysis["patches"] == sorted(chips)
        assert ("lambda" in analysis["observed"]["fits"]["X"]) == (len(chips) == 2)
        if name == "r2-d5":
            assert run.combine(rehearsal=True)["incomplete"] == [["ibm_fez-2026-10-04-r2-d5"]]
    assert main(["experiment", "analyze", "--rehearsal", "--run", str(folder)]) == 0
    assert "Rep 2 of 2026-10-04, split" in capsys.readouterr().out
    combined = run.combine(rehearsal=True)
    assert [entry["pairing"] for entry in combined["reps"]] == ["paired", "split"]
    assert combined["days"] == 1 and not combined["incomplete"]
    for entry in combined["reps"]:
        row = entry["tests"]["X lambda"]
        assert abs(row["observed"] - row["predicted"]) < 1.5 * row["bound"]
        assert entry["verdict"] == "d = 5 worse"  # Λ is far below 1 here
    pooled = combined["pooled_x"]
    sigmas = [entry["tests"]["X lambda"]["observed_uncertainty"] for entry in combined["reps"]]
    assert pooled["lambda_uncertainty"] < min(sigmas)
    assert set(combined["model"]) == {f"{b} {q}" for b in "XZ" for q in ("lambda", "eps3", "eps5")}
    assert all(row["reps"] == 2 for row in combined["model"].values())
    assert (run.RUNS / "rehearsal_combined.json").exists()
    assert main(["experiment", "combine", "--rehearsal"]) == 0
    assert "Λ over every rep" in capsys.readouterr().out


def test_a_split_reps_lambda_bound_adds_both_jobs_variances():
    from heavyhex.experiment.run import _ratio_test

    def row(value, sigma, noise, from_f, gap):
        return {
            "observed": value,
            "predicted": value,
            "observed_uncertainty": sigma,
            "simulator_noise": noise,
            "from_f": from_f,
            "calibration_gap": gap,
        }

    test = _ratio_test(row(0.05, 0.001, 0.0, 0.002, 0.0), row(0.10, 0.004, 0.0, 0.0, 0.002))
    assert test["observed"] == pytest.approx(0.5)
    expected = 0.5 * ((0.001 / 0.05) ** 2 + (0.004 / 0.1) ** 2) ** 0.5
    assert test["observed_uncertainty"] == pytest.approx(expected)
    pred = 0.5**2 * ((0.002 / 0.05) ** 2 + (0.002 / 0.1) ** 2 / 2)
    assert test["bound"] == pytest.approx(2 * (expected**2 + pred) ** 0.5)


def test_the_idle_test_waits_and_pulses_like_x_memory():
    def pulses_and_length(setting):
        circuit = native(setting)
        data = data_qubits()
        sent = Counter(circuit.find_bit(i.qubits[0]).index for i in circuit.data if i.name == "x")
        return [sent[q] for q in data], circuit.estimate_duration(TARGET, unit="s")

    idle, memory = (
        pulses_and_length(Setting("idle", "X", 2)),
        pulses_and_length(Setting("memory", "X", 2)),
    )
    assert idle[0] == memory[0]
    assert idle[1] == pytest.approx(memory[1])


def test_the_idle_test_keeps_the_h_gates_of_each_cx():
    def pulses_per_qubit(setting):
        circuit = native(setting)
        return Counter(circuit.find_bit(i.qubits[0]).index for i in circuit.data if i.name == "sx")

    assert pulses_per_qubit(Setting("idle", "X", 2)) == pulses_per_qubit(Setting("memory", "X", 2))


def test_an_offline_run_can_take_the_calibrations_gate_times():
    from heavyhex.experiment.circuits import with_durations

    timed = with_durations(TARGET, {"x": 24, "cz": 68, "measure": 1660, "reset": 1684})
    assert timed["measure"][(0,)].duration == pytest.approx(1.66e-6)
    assert timed["sx"][(0,)].duration == pytest.approx(24e-9)
    assert TARGET["measure"][(0,)].duration == pytest.approx(1.56e-6)  # left as it was


def test_a_wait_in_the_h_frame_does_not_decay():
    from qiskit import QuantumCircuit

    from heavyhex.simulation.scheduled import noisy_scheduled

    def channels(marked):
        circuit = QuantumCircuit(1)
        circuit.sx(0)  # out of |0>
        if marked:
            circuit.barrier(0, label="h frame")
        circuit.delay(250, 0, unit="dt")
        model, _ = noisy_scheduled(circuit, WORKING, TARGET.dt)
        return sum(op.name == "PAULI_CHANNEL_1" for op in model.flattened())

    assert channels(marked=False) == 1 and channels(marked=True) == 0
    idle = native(Setting("idle", "X", 1))
    marks = [i for i in idle.data if i.name == "barrier" and i.operation.label == "h frame"]
    assert marks and all(len(i.qubits) == 1 for i in marks)


def test_a_large_frequency_offset_is_unwrapped():
    from heavyhex.experiment.run import _phase

    n = np.array([1, 2, 4, 8, 16])
    r = np.exp(-0.05 * n)
    for phi in (0.1, 0.5, 1.0, -2.0):
        x, y = r * np.cos(phi * n), r * np.sin(phi * n)
        assert _phase(n, x, y, r) == pytest.approx(phi, abs=1e-6)


def test_one_f_for_every_qubit_fits_when_they_share_it():
    from heavyhex.experiment.run import _uniformity

    rng = np.random.default_rng(1)
    sigmas = np.exp(rng.uniform(np.log(0.03), np.log(0.39), 34))
    same = _uniformity(0.5 + sigmas * rng.normal(size=34), sigmas)
    spread = _uniformity(rng.uniform(0.0, 1.0, 34), sigmas)
    assert same["p_value"] > 0.01 and spread["p_value"] < 1e-6


def test_the_leakage_check_finds_a_rise_and_only_a_rise(monkeypatch):
    from heavyhex.experiment.run import _decoded, _leakage

    setting = Setting("memory", "X", 6)
    monkeypatch.setattr(run, "SETTINGS", (setting,))
    monkeypatch.setattr(run, "ROUNDS", (6,))
    monkeypatch.setattr(run, "SIMULATED_SHOTS", 40_000)
    labels = [(p, r, f"S{k}") for p in ("3", "5") for r in range(7) for k in range(4)]
    rng = np.random.default_rng(2)

    def sample(shots, rise):
        rates = np.array([0.1 + rise * r for _, r, _ in labels])
        return rng.random((shots, len(labels))) < rates

    failed = np.zeros((10_000, 2), dtype=bool)
    expected = [_decoded(setting, labels, sample(40_000, 0.0), np.zeros((40_000, 2), bool))]
    flat = _leakage([_decoded(setting, labels, sample(10_000, 0.0), failed)], expected, 10_000)
    rising = _leakage([_decoded(setting, labels, sample(10_000, 0.01), failed)], expected, 10_000)
    assert not any(row["rises"] for row in flat.values())
    assert all(row["rises"] for row in rising.values())


def test_the_agreement_bound_counts_the_measured_fs_uncertainty_and_the_calibration_gap():
    from heavyhex.experiment.run import _agreement

    def fits(lam, e3, e5, sigma=0.0):
        per = {"3": [e3, sigma], "5": [e5, sigma]}
        return {"X": {"per_round": per, "lambda": lam, "lambda_uncertainty": sigma}}

    seen, want = fits(0.60, 0.05, 0.08, 0.001), fits(0.60, 0.05, 0.08)
    plain = _agreement(seen, want, None)["tests"]["X eps3"]["bound"]
    with_f = _agreement(seen, want, None, {"X eps3": 0.05}, 0.1)["tests"]["X eps3"]["bound"]
    later = fits(0.60, 0.06, 0.08)
    with_gap = _agreement(seen, want, later)["tests"]["X eps3"]
    assert plain == pytest.approx(2 * 0.001)
    assert with_f == pytest.approx(2 * (0.001**2 + 0.005**2) ** 0.5)
    assert with_gap["bound"] == pytest.approx(2 * (0.001**2 + 0.01**2 / 2) ** 0.5)


def test_an_after_calibration_missing_a_qubit_takes_the_prepared_value():
    from heavyhex.experiment.run import _filled

    chips = {"3": fez_qubits(3, WORKING)}
    q = chips["3"][0]
    later = replace(WORKING, t1_us={**WORKING.t1_us, q: 0.0})
    fixed, gaps = _filled(later, WORKING, chips)
    assert gaps == [q] and fixed.t1_us[q] == WORKING.t1_us[q]
    assert _filled(WORKING, WORKING, chips) == (WORKING, [])


def test_submit_refuses_a_run_prepared_from_changed_code(tmp_path, capsys):
    prepared_today(tmp_path)
    info = json.loads((tmp_path / "run.json").read_text())
    info["code"] = {"commit": "x", "code": ["a", "b"], "changed": True}
    (tmp_path / "run.json").write_text(json.dumps(info))
    assert main(["experiment", "submit", "--run", str(tmp_path)]) == 2
    assert "committed code" in capsys.readouterr().err
