"""The follow-up's jobs: the circuits sent, twirled and decoupled, their cost, the caps, the
ledger, and submissions end to end against a stand-in for IBM. Nothing here contacts IBM."""

import json
from collections import Counter
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from math import ceil, isclose, pi
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("qiskit")
pytest.importorskip("qiskit_ibm_runtime")
pytest.importorskip("samplomatic")
pytest.importorskip("stim")

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from followup import submit
from followup.circuits import Setting, logical_circuit, same_action, to_stim
from followup.placement import place
from followup.submit import (
    APPROVED_SPOTS,
    JOB_SECONDS,
    JOBS,
    PER_SHOT_US,
    RUN1_CODE,
    TOTAL_SECONDS,
    Item,
    Job,
    Refused,
    approval,
    check,
    check_order,
    committed,
    decoupled,
    exp3_items,
    instances,
    job_limit,
    load,
    native,
    payload,
    program,
    twirled,
    untwirled,
    waited,
    xy4,
)
from heavyhex.experiment.circuits import with_durations
from heavyhex.patches.placement import Calibration

# IBM's calibration of ibm_kingston on 13 October 2026, from which the plan was made.
OCTOBER_13 = Calibration.load(
    Path(__file__).parent / "data" / "kingston_calibration_2026-10-13.json"
)
PLACEMENTS = place(OCTOBER_13)
PLAIN = {"twirl": True, "kerneled": True, "sampler": "executor"}


@pytest.fixture(scope="module")
def target():
    from qiskit_ibm_runtime.fake_provider import FakeKingston

    assert OCTOBER_13.durations_ns
    return with_durations(FakeKingston().target, OCTOBER_13.durations_ns)


def chip(distance):
    return PLACEMENTS[distance].qubits


def test_the_jobs_go_in_the_plans_order():
    plan = ["exp3-1", "exp3-2", "exp3-3", "exp1-1", "life-1", "exp1-2", "life-z", "life-t"]
    assert list(JOBS) == [*plan, "exp3-d"]
    order, name = [], "life-t"
    while name:
        order.append(name)
        name = JOBS[name][1]
    assert order[::-1] == plan
    assert JOBS["exp3-d"][1:] == ("life-t", "exp3-3")  # last, as exp3-3 went
    assert JOBS["life-1"][2] == JOBS["exp1-2"][2] == JOBS["life-z"][2] == "exp1-1"
    assert JOBS["life-t"][2] is None and JOBS["life-t"][0] not in submit.TWIRLED


def test_the_placement_is_the_one_approved():
    assert approval(PLACEMENTS, OCTOBER_13) == []
    assert {d: p.spot for d, p in PLACEMENTS.items()} == APPROVED_SPOTS


def test_a_new_or_worse_weak_part_waits_for_approval():
    def problems(**changes):
        calibration = OCTOBER_13
        for field, values in changes.items():
            calibration = replace(calibration, **{field: {**getattr(calibration, field), **values}})
        return approval(place(calibration), calibration)

    # A data qubit's readout is not in S, so the patches stay where they were.
    assert problems(readout={94: 0.2}) == ["data 94: readout 20.0%, not approved"]
    assert problems(readout={86: 0.25}) == ["data 86: readout 25.0%, not approved"]
    assert "data 88: T2 4 us, approved at 19 us" in problems(t2_us={88: 4.0})
    assert "coupler 86-87: CZ 20.0%, approved at 5.7%" in problems(cz={(86, 87): 0.2})
    assert "Z ancilla 78: T2 6 us, approved at 32 us" in problems(t2_us={78: 6.0})
    assert problems(t2_us={88: 15.0}) == []  # within 1.5 times
    dephased = replace(OCTOBER_13, t2_us={**OCTOBER_13.t2_us, 70: 40.0})
    assert approval(PLACEMENTS, dephased) == ["d=3 data 70: T2 40 us, below 50 us"]


def test_h_and_cx_are_written_as_run_1_wrote_them():
    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    names = [(i.operation.name, [qc.find_bit(q).index for q in i.qubits]) for i in native(qc).data]
    h = lambda q: [("rz", [q]), ("sx", [q]), ("rz", [q])]  # noqa: E731
    assert names == h(0) + h(1) + [("cz", [0, 1])] + h(1)
    assert Operator(native(qc)).equiv(Operator(qc))


def test_xy4_is_the_identity_with_y_a_turned_x():
    gates, spacing = xy4()
    qc = QuantumCircuit(1)
    for gate in gates:
        qc.append(gate, [0])
    assert Operator(qc).equiv(Operator(QuantumCircuit(1)))
    assert [g.name for g in gates].count("x") == 4 and {g.name for g in gates} == {"x", "rz"}
    assert len(spacing) == len(gates) + 1 and isclose(sum(spacing), 1)


@pytest.mark.parametrize("basis", "XZ")
def test_twirled_memory_acts_as_its_logical_circuit_in_every_twirl(basis, target):
    item = twirled(Setting("memory", basis, 3, 2), chip(3), target)
    assert (item.randomizations, item.shots, item.executions) == (16, 256, 4096)
    check(item, randomizations=4, seed=1)
    drawn = instances(item, randomizations=2, seed=1)
    assert any(  # two draws bind different angles
        a.operation.params != b.operation.params
        for a, b in zip(drawn[0].data, drawn[1].data, strict=True)
    )


def test_a_dropped_xy4_pulse_is_caught(target):
    """A pulse whose Pauli commutes with every detector leaves them quiet; the flows don't."""
    item = twirled(Setting("memory", "Z", 3, 1), chip(3), target)
    (bound,) = instances(item, 1)
    logical, _ = logical_circuit(Setting("memory", "Z", 3, 1), item.chip, target.num_qubits)
    assert same_action(bound, logical)
    data = set(PLACEMENTS[3].roles["data"])
    pulses = [
        k
        for k, i in enumerate(bound.data)
        if i.operation.name == "x" and bound.find_bit(i.qubits[0]).index in data
    ]
    for k in pulses[:4]:
        dropped = bound.copy_empty_like()
        for j, i in enumerate(bound.data):
            if j != k:
                dropped.append(i)
        assert not same_action(dropped, logical)
    broken = Item(item.label, dropped, item.seconds, 1, setting=item.setting, chip=item.chip)
    with pytest.raises(ValueError, match="does not act"):
        check(broken)


def test_untwirled_memory_goes_in_two_pieces_and_acts_as_its_logical_circuit(target):
    pieces = untwirled(Setting("memory", "Z", 3, 2), chip(3), target)
    assert len(pieces) == 2 and pieces[0] is pieces[1]
    assert pieces[0].samplex is None and sum(p.executions for p in pieces) == 4096
    check(pieces[0])
    assert not any(i.operation.name == "box" for i in pieces[0].circuit.data)


def test_the_idle_test_waits_where_memory_has_its_czs(target):
    memory = twirled(Setting("memory", "X", 5, 1), chip(5), target)
    idle_x, idle_y = (
        twirled(Setting("idle", "X", 5, 1, readout), chip(5), target) for readout in "XY"
    )
    assert idle_x.seconds == idle_y.seconds == memory.seconds
    assert (idle_x.randomizations, idle_x.executions) == (8, 2048)
    wires = submit._wires(waited(memory.circuit, target))
    assert submit._wires(idle_x.circuit) == wires == submit._wires(idle_y.circuit)
    assert not any(i.operation.name == "cz" for i in idle_x.circuit.data)
    check(idle_x, randomizations=2)  # the data read 0 in X ...
    check(idle_y, randomizations=1)  # ... and both act as the logical idle test
    (bound,) = instances(idle_y, randomizations=1)
    model, records = to_stim(bound)  # ... and Y reads at random
    data = [k for (name, _), k in records.items() if name == "d5"]
    assert 0.3 < model.compile_sampler(seed=0).sample(256)[:, data].mean() < 0.7


def test_decoupling_fills_each_wait_but_the_first(target):
    qubits = chip(3)
    logical, _ = logical_circuit(Setting("memory", "X", 3, 2), qubits, target.num_qubits)
    circuit = decoupled(native(logical), target, qubits)
    pulses, first_gate = Counter(), {}
    for k, item in enumerate(circuit.data):
        for q in (circuit.find_bit(b).index for b in item.qubits):
            assert q in qubits, "nothing on qubits the patch doesn't use"
            name = item.operation.name
            if name not in ("delay", "barrier", "rz"):
                first_gate.setdefault(q, (k, name))
            if name == "x":
                pulses[q] += 1
    assert all(name != "x" for _, name in first_gate.values())  # no XY4 before first use
    assert all(pulses[q] % 4 == 0 and pulses[q] for q in PLACEMENTS[3].roles["data"])
    plain = decoupled(native(logical), target, qubits, decouple=False)
    assert "x" not in plain.count_ops()
    assert isclose(plain.estimate_duration(target), circuit.estimate_duration(target))


def test_a_gate_the_device_lacks_is_refused(target):
    qc = QuantumCircuit(target.num_qubits)
    qc.cz(0, 50)
    with pytest.raises(ValueError, match="not native"):
        submit._item("far", qc, target, 1)


def test_exp3_sends_the_arm_in_pieces_of_the_references_shots(target):
    from followup.leakage_control import select

    items = exp3_items("A", select(PLACEMENTS[3].layout), target)
    assert [i.label for i in items] == ["R0", "R1", "A", "A", "A"]
    assert {i.shots for i in items} == {1000} and sum(i.executions for i in items) == 5000
    assert isclose(items[-1].seconds, 30 * 4.864e-6)


@pytest.fixture(scope="module")
def jobs(target):
    return submit.build(list(JOBS), PLACEMENTS, target)


def test_the_cost_is_ibms_billing_rule_and_its_bound_loads_in_parts(target, jobs):
    from followup.leakage_control import select

    job = Job("exp3-1", exp3_items("A", select(PLACEMENTS[3].layout), target), "kerneled")
    expected = 2.1 + sum(i.executions * (i.seconds + PER_SHOT_US * 1e-6) for i in job.items)
    assert isclose(job.seconds, expected) and job.billed == ceil(expected) == 4
    assert job.instances == 5 and job.bound == job.seconds
    assert PER_SHOT_US == 253
    exp1 = jobs["exp1-1"]
    assert exp1.instances == 16 * 16 + 10 * 8  # two parts of 300
    assert isclose(exp1.bound, exp1.seconds + 2.1)
    assert ceil(exp1.bound) <= JOB_SECONDS
    life = jobs["life-1"]
    assert life.instances == 10 * 4 + 12 * 4 and life.bound == life.seconds
    assert life.billed == 10 <= submit.SECONDS["life-1"] == 14
    life_z, life_t = jobs["life-z"], jobs["life-t"]
    assert [(i.label, i.executions) for i in life_z.items] == [
        (i.label, i.executions) for i in life.items[:10]
    ]  # life-1's idle test read in Z, as it went
    assert life_z.twirled and life_z.instances == 10 * 4 and life_z.billed == 6
    assert not life_t.twirled and life_t.instances == 12 * 2 and life_t.billed == 5
    for job in (life_z, life_t):  # IBM cancels each at its own cap
        assert job.bound == job.seconds
        assert ceil(1.5 * job.bound) + 2 > submit.SECONDS[job.name] == 7
    planned = sum(job.billed for name, job in jobs.items() if name not in submit.OPTIONAL)
    assert planned == 87 <= TOTAL_SECONDS  # life-1 as IBM charged it
    assert planned + jobs["exp3-d"].billed > submit.PLAN_SECONDS  # so exp3-d can't go


def stub(name="exp1-1", billed=27):
    """A job that costs `billed` s by the billing rule, in one part."""
    seconds = (billed - 2.1 - 0.5) / 1000 - PER_SHOT_US * 1e-6
    return Job(name, (Item("x", QuantumCircuit(1), seconds, 1000),))


def test_the_caps_refuse_a_dear_job_an_overspent_total_and_a_short_counter():
    assert stub(billed=27).billed == 27
    assert job_limit(stub(), [], 375) == JOB_SECONDS
    assert job_limit(stub(billed=4), [], 375) == ceil(1.5 * stub(billed=4).bound) + 2 == 8
    with pytest.raises(Refused, match="over its 30 s"):
        job_limit(stub(billed=31), [], 375)
    assert job_limit(stub("life-1", billed=10), [], 375) == 14  # its own cap, not 17
    with pytest.raises(Refused, match="over its 14 s"):
        job_limit(stub("life-1", billed=15), [], 375)
    assert job_limit(stub("life-z", billed=6), [], 375) == 7  # not 11
    with pytest.raises(Refused, match="over its 7 s"):
        job_limit(stub("life-t", billed=8), [], 375)
    spent = [{"job": "exp3-1", "status": "sent", "limit_s": 30, "usage": None}] * 2
    assert committed(spent) == 60
    assert job_limit(stub(billed=27), spent[:1], 375) == 30
    with pytest.raises(Refused, match="90 s approved"):
        job_limit(stub(billed=30), [*spent, {**spent[0], "limit_s": 5}], 375)
    with pytest.raises(Refused, match="counter"):
        job_limit(stub(billed=27), [], 36)
    with pytest.raises(Refused, match="counter"):
        job_limit(stub(billed=27), [], None)
    with pytest.raises(Refused, match="below 285 s"):  # used elsewhere, so the floor holds it
        job_limit(stub(billed=27), [], 300)
    assert job_limit(stub(billed=4), [*spent, {**spent[0], "limit_s": 25}], 375) == 5


def test_life_z_and_life_t_fit_the_90_s_once_ibm_has_counted_life_1(jobs):
    """The ledger as it stands after exp1-2: life-1 counts at its 14 s limit until IBM has
    counted its 10 s, and only then does life-t fit after life-z."""

    def spent(job, charged=None, limit=30):
        usage = {"status": "pending"}
        if charged is not None:
            usage = {"status": "complete", "qpu_charge_time_seconds": charged}
        return {"job": job, "status": "done", "limit_s": limit, "usage": usage}

    before = [spent("exp3-1", 4), spent("exp3-2", 4), spent("exp3-3", 4), spent("exp1-1", 27),
              spent("exp1-2", 27)]  # fmt: skip
    pending, counted = [*before, spent("life-1", limit=14)], [*before, spent("life-1", 10, 14)]
    life_z, life_t = jobs["life-z"], jobs["life-t"]
    assert committed(pending) == 80 and committed(counted) == 76
    assert job_limit(life_z, pending, 375) == job_limit(life_z, counted, 299) == 7
    with pytest.raises(Refused, match="90 s approved"):
        job_limit(life_t, [*pending, spent("life-z", 6, 7)], 375)  # 86 + 5
    assert job_limit(life_t, [*counted, spent("life-z", limit=7)], 375) == 7  # 83 + 5
    assert job_limit(life_t, [*counted, spent("life-z", 6, 7)], 293) == 7
    with pytest.raises(Refused, match="below 285 s"):
        job_limit(life_t, [*counted, spent("life-z", 6, 7)], 291)


def test_an_optional_job_stays_within_the_plans_80_s_and_no_cap_passes_30_s(monkeypatch):
    six = [{"job": "exp1-1", "status": "done", "limit_s": 30,
            "usage": {"status": "complete", "qpu_charge_time_seconds": s}}
           for s in (4, 4, 4, 27, 10, 27)]  # fmt: skip
    with pytest.raises(Refused, match=r"optional.*the plan's 80 s"):
        job_limit(stub("exp3-d", billed=5), six, 375)  # 76 + 5
    assert job_limit(stub("exp3-d", billed=4), six, 375) == 8  # 76 + 4
    assert job_limit(stub("exp1-2", billed=5), six, 375) == 9  # only optional jobs
    monkeypatch.setitem(submit.SECONDS, "life-1", 45)
    assert submit.job_cap("life-1") == JOB_SECONDS


def test_the_ledger_counts_ibms_charge_once_final_and_the_limit_till_then():
    attempts = [
        {"job": "exp3-1", "status": "refused", "limit_s": 30},
        {"job": "exp3-1", "status": "failed", "limit_s": 8,
         "usage": {"status": "complete", "qpu_charge_time_seconds": 1}},
        {"job": "exp3-1", "status": "done", "limit_s": 30,
         "usage": {"status": "complete", "qpu_charge_time_seconds": 4}},
        {"job": "exp3-2", "status": "done", "limit_s": 30, "usage": {"status": "pending"}},
        {"job": "exp3-3", "status": "claimed", "limit_s": 22, "usage": None},
    ]  # fmt: skip
    assert committed(attempts) == 1 + 4 + 30 + 22
    assert TOTAL_SECONDS == 90


def attempt(job, status="done", passed=True, **flags):
    return {
        "job": job, "status": status, "folder": f"f-{job}", "flags": {**PLAIN, **flags},
        "checks": {"passed": passed, "failed": [] if passed else ["A: IQ points repeat"]},
        "billed_s": 4, "usage": None,
    }  # fmt: skip


def test_each_job_goes_once_and_after_the_one_before_it():
    with pytest.raises(Refused, match="after exp3-1"):
        check_order("exp3-2", [], PLAIN)
    check_order("exp3-1", [{**attempt("exp3-1"), "status": "refused"}], PLAIN)
    with pytest.raises(Refused, match="out already"):
        check_order("exp3-1", [attempt("exp3-1", "claimed")], PLAIN)
    with pytest.raises(Refused, match="gone already"):
        check_order("exp3-1", [attempt("exp3-1")], PLAIN)
    with pytest.raises(Refused, match="after exp3-1"):
        check_order("exp3-2", [attempt("exp3-1", "sent")], PLAIN)
    check_order("exp3-2", [attempt("exp3-1")], PLAIN)


def test_a_failed_job_goes_again_only_with_a_fallback_and_the_next_goes_alike():
    failed = attempt("exp3-1", "failed")
    with pytest.raises(Refused, match="failed so already"):
        check_order("exp3-1", [failed], PLAIN)
    with pytest.raises(Refused, match="failed so already"):  # twirling is not Exp 3's
        check_order("exp3-1", [failed], {**PLAIN, "twirl": False})
    check_order("exp3-1", [failed], {**PLAIN, "kerneled": False})
    resent = attempt("exp3-1", kerneled=False)
    with pytest.raises(Refused, match="goes as exp3-1 went, with --classified"):
        check_order("exp3-2", [failed, resent], PLAIN)
    check_order("exp3-2", [failed, resent], {**PLAIN, "kerneled": False})
    check_order("exp1-1", [attempt("exp3-3", kerneled=False)], PLAIN)  # another experiment
    with pytest.raises(Refused, match="goes as exp1-1 went, without --untwirled"):
        check_order("life-1", [attempt("exp1-1")], {**PLAIN, "twirl": False})
    with pytest.raises(Refused, match="after life-1"):
        check_order("exp1-2", [attempt("exp1-1")], PLAIN)
    with pytest.raises(Refused, match="after life-1"):  # skipped by hand: still not
        check_order("exp1-2", [attempt("exp1-1")], PLAIN, despite=True)
    # exp1-1 went untwirled, and so did life-1, again with run 1's sampler.
    untwirled = [attempt("exp1-1", "failed"), attempt("exp1-1", twirl=False)]
    went = [*untwirled, attempt("life-1", "failed", twirl=False)]
    check_order("life-1", went, {**PLAIN, "twirl": False, "sampler": "legacy"})
    went.append(attempt("life-1", twirl=False, sampler="legacy"))
    check_order("exp1-2", went, {**PLAIN, "twirl": False})
    with pytest.raises(Refused, match="goes as exp1-1 went, with --untwirled"):
        check_order("exp1-2", went, PLAIN)


def test_life_1_ended_by_ibm_after_a_twirled_exp1_1_holds_exp1_2_only_till_passed_by_hand():
    ended = {**attempt("life-1", "failed"), "reason": "CANCELLED: over its 14 s"}
    went = [attempt("exp1-1"), ended]
    for flags in (PLAIN, {**PLAIN, "twirl": False}, {**PLAIN, "twirl": False, "sampler": "legacy"}):
        with pytest.raises(Refused, match=r"failed so already|goes as exp1-1 went"):
            check_order("life-1", went, flags)  # no fallback is left
    with pytest.raises(Refused, match=r"life-1 was ended by IBM \(CANCELLED: over its 14 s\)"):
        check_order("exp1-2", went, PLAIN)
    check_order("exp1-2", went, PLAIN, despite=True)
    with pytest.raises(Refused, match="goes as exp1-1 went"):
        check_order("exp1-2", went, {**PLAIN, "twirl": False}, despite=True)
    with pytest.raises(Refused, match="after life-1"):  # while a life-1 is out, never
        check_order("exp1-2", [*went, attempt("life-1", "sent")], PLAIN, despite=True)


def test_life_z_goes_after_exp1_2_as_exp1_1_went_and_life_t_after_it():
    went = [attempt("exp1-1"), {**attempt("life-1", "failed"), "reason": "CANCELLED"}]
    with pytest.raises(Refused, match="after exp1-2"):
        check_order("life-z", went, PLAIN)
    went.append(attempt("exp1-2"))
    with pytest.raises(Refused, match="goes as exp1-1 went, without --untwirled"):
        check_order("life-z", went, {**PLAIN, "twirl": False})
    check_order("life-z", went, PLAIN)
    legacy = {**PLAIN, "sampler": "legacy"}
    with pytest.raises(Refused, match="after life-z"):
        check_order("life-t", [*went, attempt("life-z", "sent")], legacy)
    went.append(attempt("life-z"))
    check_order("life-t", went, legacy)
    with pytest.raises(Refused, match="after life-t"):
        check_order("exp3-d", [*went, attempt("exp3-3")], PLAIN)


def test_failed_checks_or_an_overcharge_hold_the_next_job():
    bad = attempt("exp3-1", passed=False)
    with pytest.raises(Refused, match=r"failed its checks.*IQ points repeat"):
        check_order("exp3-2", [bad], PLAIN)
    check_order("exp3-2", [bad], PLAIN, despite=True)
    check_order("exp3-1", [bad], {**PLAIN, "kerneled": False})  # its data failed: again
    dear = {**attempt("exp1-1"), "billed_s": 27,
            "usage": {"status": "complete", "qpu_charge_time_seconds": 29}}  # fmt: skip
    with pytest.raises(Refused, match="charged 29 s"):
        check_order("life-1", [dear], PLAIN)


def test_the_executor_gets_one_item_a_circuit_and_nothing_is_sent(target):
    items = (
        twirled(Setting("memory", "X", 3, 1), chip(3), target),
        twirled(Setting("idle", "X", 5, 1), chip(5), target),
    )
    job = Job("exp1-1", items)
    sent, options = program(job)
    assert sent.shots == 256 and [i.shape for i in sent.items] == [(16,), (8,)]
    assert options.execution.rep_delay == 250e-6 and options.execution.scheduler_timing
    encoded = payload(job)
    assert encoded["schema_version"] == "v2.0" and len(encoded["quantum_program"]["items"]) == 2


def test_load_flattens_twirls_and_undoes_readout_flips(tmp_path):
    rng = np.random.default_rng(0)
    bits = rng.random((2, 3, 4)) < 0.5
    flips = rng.random((2, 1, 4)) < 0.5
    iq = rng.normal(size=(5, 4)) + 1j * rng.normal(size=(5, 4))
    np.savez_compressed(
        tmp_path / "results.npz",
        **{"0/m5": bits ^ flips, "0/measurement_flips.m5": flips, "1/m_iq": iq, "2/m_iq": iq},
    )
    items = [{"label": "memory"}, {"label": "A"}, {"label": "A"}]
    (tmp_path / "run.json").write_text(json.dumps({"items": items}))
    found = load(tmp_path)
    assert np.array_equal(found["memory"]["m5"], bits.reshape(6, 4))
    assert found["A"]["m"].shape == (10, 4) and np.iscomplexobj(found["A"]["m"])


def test_ibm_failures_are_told_apart_from_refusals():
    from qiskit_ibm_runtime.api.exceptions import RequestsApiError
    from qiskit_ibm_runtime.exceptions import IBMInputValueError, IBMRuntimeError

    def raised(status):
        try:
            try:
                raise RequestsApiError("no", status)
            except RequestsApiError:
                raise IBMRuntimeError("Failed to run program") from None
        except IBMRuntimeError as error:
            return error

    assert submit._never_sent(raised(400)) and submit._never_sent(raised(403))
    assert not submit._never_sent(raised(500)) and not submit._never_sent(raised(-1))
    assert submit._never_sent(IBMInputValueError("bad options"))
    assert not submit._never_sent(ConnectionError("dropped"))


# Submissions against a stand-in for IBM


@pytest.fixture
def ibm(tmp_path, monkeypatch, target):
    """IBM replaced: the device, its counter, its jobs; every job sent is recorded."""
    import heavyhex.experiment.run as run1

    monkeypatch.setattr(submit, "RUNS", tmp_path)
    monkeypatch.setattr(submit, "LEDGER", tmp_path / "followup" / "ledger.json")
    monkeypatch.setattr(submit, "FIND_EVERY_S", 0)
    monkeypatch.setattr(submit, "device_calibration", lambda device: OCTOBER_13)
    monkeypatch.setattr(
        submit,
        "provenance",
        lambda: {"commit": "c", "code": list(RUN1_CODE), "followup": "f", "changed": False},
    )
    state = SimpleNamespace(
        sent=[], jobs={}, listed=[], outcome="DONE", lost=False, iq=None, left=375
    )

    def finish(folder, job):
        record = json.loads((folder / "job.json").read_text())
        record |= {"status": job.status(), "usage": {"status": "complete",
                                                      "qpu_charge_time_seconds": 4}}  # fmt: skip
        (folder / "job.json").write_text(json.dumps(record))

    monkeypatch.setattr(run1, "finish", finish)
    rng = np.random.default_rng(1)

    def result(job):
        from qiskit_ibm_runtime.results.quantum_program import (
            ItemMetadata,
            QuantumProgramItemResult,
            QuantumProgramResult,
            SchedulerTiming,
        )

        if state.outcome != "DONE":
            raise RuntimeError(state.outcome)
        out = []
        for item in job.items:
            bits = item.circuit.cregs[0].size
            iq = rng.normal(size=(item.shots, bits)) + 1j * rng.normal(size=(item.shots, bits))
            if state.iq == "repeated":
                iq[:, 1] = iq[:, 0]
            timing = SchedulerTiming("", round(item.seconds / target.dt))
            out.append(QuantumProgramItemResult({"m_iq": iq}, ItemMetadata(timing)))
        return QuantumProgramResult(out)

    def send(backend, job, limit, tags, sampler="executor"):
        state.sent.append((job.name, limit, tags))
        handle = SimpleNamespace(
            job_id=lambda: f"id{len(state.sent)}",
            status=lambda: state.outcome if state.outcome != "dropped" else "RUNNING",
            usage_estimation={"quantum_seconds": 4},
            result=lambda: result(job),
            metrics=lambda: {},
            tags=tags,
        )
        state.jobs[handle.job_id()] = handle
        if state.lost:
            state.listed.append(handle)
            raise ConnectionError("the reply was lost")
        return handle

    monkeypatch.setattr(submit, "send", send)
    state.service = SimpleNamespace(
        backend=lambda name: SimpleNamespace(name=name, target=target),
        usage=lambda: {"usage_remaining_seconds": state.left},
        job=lambda job_id: state.jobs[job_id],
        jobs=lambda **_: list(state.listed),
    )
    return state


def test_a_submission_end_to_end_against_a_stand_in(tmp_path, ibm, target):
    """exp3-1 through every check, the ledger and the record, with IBM replaced."""
    with pytest.raises(Refused, match="after exp3-2"):
        submit.submit("exp3-3", service=ibm.service)
    folder = submit.submit("exp3-1", service=ibm.service)
    ((name, limit, tags),) = ibm.sent
    assert (name, limit) == ("exp3-1", 8) and tags == [submit.TAG, f"heavyhex:{folder.name}"]
    with pytest.raises(Refused, match="gone already"):
        submit.submit("exp3-1", service=ibm.service)
    assert len(ibm.sent) == 1
    (attempt,) = submit.ledger(tmp_path / "followup" / "ledger.json")
    assert attempt["status"] == "done" and committed([attempt]) == 4
    record = json.loads((folder / "job.json").read_text())
    assert record["code"] == {"commit": "c", "code": list(RUN1_CODE), "followup": "f",
                              "changed": False}  # fmt: skip
    checks = record["checks"]
    assert checks["passed"] and checks["failed"] == []
    assert checks["circuits"]["A"]["m"] == {"shots": 3000, "bits": 300, "iq_distinct_per_shot": 1.0}
    assert checks["schedule_gap_dt"] == [0, 0]
    assert {p.name for p in folder.iterdir()} >= {
        "job.json", "run.json", "calibration.json", "circuits.qpy.gz", "program.json.gz",
        "results.npz", "schedule.json.gz",
    }  # fmt: skip
    data = load(folder)
    assert data["R0"]["m"].shape == (1000, 10) and data["A"]["m"].shape == (3000, 300)
    assert not (tmp_path / "followup" / "ledger.lock").exists()


def test_life_1_and_exp1_2_go_with_their_own_limits_against_a_stand_in(ibm, jobs, monkeypatch):
    """After exp1-1, through submit() with IBM replaced, which refuses each job it gets: life-1
    goes with its own 14 s, the counter's floor holds it, exp1-2 goes with 30 s after it, and
    after a life-1 IBM ended only by hand; then life-z and life-t with 7 s each."""
    monkeypatch.setattr(submit, "build", lambda names, *_, **__: {n: jobs[n] for n in names})
    monkeypatch.setattr(submit, "check", lambda item: None)  # each circuit is checked above
    monkeypatch.setattr(submit, "payload", lambda job: {})
    tried = []

    def send(backend, job, limit, tags, sampler="executor"):
        tried.append((job.name, limit))
        raise Refused("the stand-in refuses every job")

    monkeypatch.setattr(submit, "send", send)

    def done(job, charged, **fields):
        usage = {"status": "complete", "qpu_charge_time_seconds": charged}
        return {**attempt(job), "tags": [submit.TAG, f"heavyhex:f-{job}"], "id": None,
                "billed_s": charged, "limit_s": 30, "usage": usage, **fields}  # fmt: skip

    before = [done("exp3-1", 4), done("exp3-2", 4), done("exp3-3", 4), done("exp1-1", 27)]

    def go(name, attempts, left=375, **flags):
        submit._save_ledger(attempts)
        ibm.left = left
        with pytest.raises(Refused) as refusal:
            submit.submit(name, service=ibm.service, **flags)
        return str(refusal.value)

    assert "the stand-in refuses" in go("life-1", before)
    assert tried[-1] == ("life-1", 14)
    assert "below 285 s" in go("life-1", before, left=298) and len(tried) == 1
    go("life-1", before, left=299)
    assert tried[-1] == ("life-1", 14)
    assert "after life-1" in go("exp1-2", before)
    go("exp1-2", [*before, done("life-1", 10)])
    assert tried[-1] == ("exp1-2", 30)
    ended = done("life-1", 14, status="failed", reason="CANCELLED")
    assert "life-1 was ended by IBM" in go("exp1-2", [*before, ended]) and len(tried) == 3
    assert "failed so already" in go("life-1", [*before, ended])
    assert "goes as exp1-1 went" in go("life-1", [*before, ended], twirl=False)
    go("exp1-2", [*before, ended], despite=True)
    assert tried[-1] == ("exp1-2", 30) and len(tried) == 4
    # Then life-z, twirled, and life-t through run 1's sampler, each with its own 7 s.
    before += [done("life-1", 10, status="failed", reason="CANCELLED"), done("exp1-2", 27)]
    go("life-z", before)
    assert tried[-1] == ("life-z", 7)
    assert "untwirled" in go("life-z", before, sampler="legacy")
    assert "after life-z" in go("life-t", before, sampler="legacy") and len(tried) == 5
    go("life-t", [*before, done("life-z", 6)], sampler="legacy")
    assert tried[-1] == ("life-t", 7)


def test_malformed_iq_data_hold_the_next_job_until_it_goes_classified(ibm):
    ibm.iq = "repeated"
    folder = submit.submit("exp3-1", service=ibm.service)
    record = json.loads((folder / "job.json").read_text())
    assert not record["checks"]["passed"]
    with pytest.raises(Refused, match="failed its checks"):
        submit.submit("exp3-2", service=ibm.service)
    with pytest.raises(Refused, match="failed so already"):
        submit.submit("exp3-1", service=ibm.service)
    ibm.iq = None
    again = submit.submit("exp3-1", kerneled=False, service=ibm.service)
    assert again.name.endswith("exp3-1-classified")


def test_a_job_ibm_refuses_counts_nothing_and_may_go_again(tmp_path, ibm, monkeypatch):
    from qiskit_ibm_runtime.api.exceptions import RequestsApiError
    from qiskit_ibm_runtime.exceptions import IBMRuntimeError

    def send(*_, **__):
        try:
            raise RequestsApiError("not on this plan", 400)
        except RequestsApiError:
            raise IBMRuntimeError("Failed to run program: not on this plan") from None

    monkeypatch.setattr(submit, "send", send)
    with pytest.raises(Refused, match=r"not on this plan.*may go again"):
        submit.submit("exp3-1", service=ibm.service)
    (attempt,) = submit.ledger(tmp_path / "followup" / "ledger.json")
    assert attempt["status"] == "refused" and committed([attempt]) == 0
    assert not any(p.is_dir() for p in tmp_path.iterdir() if p.name != "followup")
    check_order("exp3-1", [attempt], PLAIN)


def test_a_send_whose_reply_is_lost_is_found_by_its_tag(ibm):
    ibm.lost = True
    folder = submit.submit("exp3-1", service=ibm.service)
    (attempt,) = submit.ledger(submit.LEDGER)
    assert attempt["status"] == "done" and attempt["id"] == "id1"
    assert folder.name.endswith("exp3-1") and len(ibm.sent) == 1


def test_an_unlisted_send_counts_at_its_limit_and_is_released_later(ibm):
    ibm.lost = True
    ibm.service.jobs = lambda **_: []  # IBM never lists it
    with pytest.raises(Refused, match="counts at its 8 s limit"):
        submit.submit("exp3-1", service=ibm.service)
    (attempt,) = submit.ledger(submit.LEDGER)
    assert attempt["status"] == "claimed" and committed([attempt]) == 8
    with pytest.raises(Refused, match="out already"):
        submit.submit("exp3-1", service=ibm.service)
    with pytest.raises(Refused, match="doesn't list exp3-1 yet"):
        submit.fetch("exp3-1", service=ibm.service)
    attempts = submit.ledger(submit.LEDGER)
    old = datetime.now(timezone.utc) - timedelta(minutes=16)
    attempts[0]["claimed_utc"] = old.isoformat(timespec="seconds")
    submit._save_ledger(attempts)
    with pytest.raises(Refused, match="released"):
        submit.fetch("exp3-1", service=ibm.service)
    (attempt,) = submit.ledger(submit.LEDGER)
    assert attempt["status"] == "refused" and committed([attempt]) == 0


def test_a_dropped_wait_leaves_the_job_out_for_fetch(ibm):
    ibm.outcome = "dropped"
    with pytest.raises(Refused, match="--fetch exp3-1"):
        submit.submit("exp3-1", service=ibm.service)
    (attempt,) = submit.ledger(submit.LEDGER)
    assert attempt["status"] == "sent" and attempt["id"] == "id1"
    ibm.outcome = "DONE"
    folder = submit.fetch("exp3-1", service=ibm.service)
    (attempt,) = submit.ledger(submit.LEDGER)
    assert attempt["status"] == "done" and (folder / "results.npz").exists()
    assert len(ibm.sent) == 1


def test_a_job_ibm_fails_counts_its_charge_and_goes_again_with_a_fallback(ibm):
    ibm.outcome = "ERROR"
    with pytest.raises(Refused, match="ERROR"):
        submit.submit("exp3-1", service=ibm.service)
    (attempt,) = submit.ledger(submit.LEDGER)
    assert attempt["status"] == "failed" and committed([attempt]) == 4
    with pytest.raises(Refused, match="failed so already"):
        submit.submit("exp3-1", service=ibm.service)
    ibm.outcome = "DONE"
    folder = submit.submit("exp3-1", kerneled=False, service=ibm.service)
    assert folder.name.endswith("-classified")


def test_a_job_ibm_lists_but_the_ledger_misses_stops_everything(ibm):
    ibm.service.jobs = lambda **_: [SimpleNamespace(job_id=lambda: "elsewhere", tags=[])]
    with pytest.raises(Refused, match="elsewhere"):
        submit.submit("exp3-1", service=ibm.service)
    assert ibm.sent == []


def test_uncommitted_code_or_a_moved_placement_sends_nothing(ibm, monkeypatch):
    monkeypatch.setattr(submit, "send", lambda *_, **__: pytest.fail("sent"))
    code = {"commit": "c", "code": list(RUN1_CODE), "followup": "f"}
    monkeypatch.setattr(submit, "provenance", lambda: {**code, "changed": True})
    with pytest.raises(Refused, match="commit the code"):
        submit.submit("exp3-1", service=ibm.service)
    monkeypatch.setattr(
        submit, "provenance", lambda: {**code, "code": ["x", "y"], "changed": False}
    )
    with pytest.raises(Refused, match="no longer run 1's"):
        submit.submit("exp3-1", service=ibm.service)
    moved = replace(OCTOBER_13, t2_us={**OCTOBER_13.t2_us, 52: 10.0})
    monkeypatch.setattr(submit, "device_calibration", lambda device: moved)
    with pytest.raises(Refused, match="approved"):
        submit.submit("exp3-1", service=ibm.service)
    with pytest.raises(Refused, match="untwirled"):
        submit.submit("exp1-1", service=ibm.service, sampler="legacy")


def test_the_command_never_sends_from_a_dry_run(monkeypatch, capsys):
    monkeypatch.setattr(submit, "submit", lambda *_, **__: pytest.fail("sent"))
    with pytest.raises(SystemExit):
        submit.main(["--dry", "--submit", "exp3-1"])
    with pytest.raises(SystemExit):
        submit.main(["--calibration", "saved.json", "--submit", "exp3-1"])
    capsys.readouterr()


def test_twirls_never_flip_readouts(target):
    item = twirled(Setting("memory", "Z", 3, 1), chip(3), target)
    assert item.samplex is not None
    drawn = item.samplex.sample(item.samplex.inputs().bind(), num_randomizations=1)
    assert not any(key.startswith(submit.FLIPS) for key in drawn)
    values = np.asarray(drawn["parameter_values"], dtype=float)
    assert np.allclose(values, np.round(values / (pi / 2)) * pi / 2, atol=1e-5)
