"""The 65-qubit d=5 flagged circuit, checked on Aer."""

import json
from itertools import combinations
from pathlib import Path

import pytest

from heavyhex.circuits.flagged import memory_circuit_flagged, propagate_fault, single_faults
from heavyhex.core import Pauli
from heavyhex.patches.operators import D5

pytest.importorskip("qiskit_aer")

from heavyhex.simulation.aer import run_flagged_circuit, run_memory_flagged  # noqa: E402


def test_d5_gates_follow_every_blueprint_bond():
    saved = json.loads(
        (Path(__file__).resolve().parents[1] / "docs/figures/d5_fez_layout.json").read_text()
    )
    circuit, schedule = memory_circuit_flagged(D5, rounds=3)
    roles = schedule.roles
    mapping = {int(q) - 1: physical for q, physical in saved["data_qubits"].items()}
    mapping.update({q: saved["x_gauge_ancillas"][name] for name, q in roles.x_ancillas.items()})
    mapping.update({q: saved["z_gauge_ancillas"][name] for name, q in roles.z_ancillas.items()})
    bonds = {frozenset((a, b)) for _, a, b in saved["couplings"]}
    for name, arms in roles.z2_arms.items():
        for data, relay in arms:
            candidates = [
                q
                for q in saved["boundary_relays"]
                if frozenset((mapping[data], q)) in bonds
                and frozenset((q, saved["z_gauge_ancillas"][name])) in bonds
            ]
            assert len(candidates) == 1
            mapping[relay] = candidates[0]
    assert set(mapping) == set(range(65))
    assert len(set(mapping.values())) == 65
    used = set()
    for instruction in circuit.data:
        if instruction.operation.name == "cx":
            edge = frozenset(mapping[circuit.find_bit(q).index] for q in instruction.qubits)
            assert edge in bonds
            used.add(edge)
    assert used == bonds
    assert circuit.num_qubits == 65
    assert len(schedule.measurements) == 20 + 3 * 56
    assert [m.bit for m in schedule.measurements] == list(range(188))


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_noiseless_d5_has_quiet_relays_and_no_detectors(basis):
    circuit, schedule = memory_circuit_flagged(D5, rounds=3, basis=basis)
    relays = [m.bit for m in schedule.measurements if m.kind == "relay"]
    for record in run_flagged_circuit(circuit, schedule, shots=64, seed=27):
        assert record["success"] is None  # lookup can't grade d=5
        assert not any(record["detectors"])
        assert all(record["gauge_bits"][bit] == 0 for bit in relays)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_all_d5_single_data_paulis_have_expected_syndrome(basis):
    for q in D5.data_qubits:
        x, z = Pauli.x_on((q,)), Pauli.z_on((q,))
        for error in (x, z, x * z):
            records = run_memory_flagged(D5, basis=basis, error=error, shots=8, seed=31)
            assert all(r["syndrome"] == D5.code.syndrome(error) for r in records), (q, error)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_d5_gadget_faults_do_not_grow_or_hide_logicals(basis):
    circuit, schedule = memory_circuit_flagged(D5, basis=basis)
    roles = schedule.roles
    flags = set(roles.x_ancillas.values())
    relays = {q for arms in roles.z2_arms.values() for _, q in arms}
    candidates = [Pauli()]
    for q in D5.data_qubits:
        x, z = Pauli.x_on((q,)), Pauli.z_on((q,))
        candidates.extend((x, z, x * z))
    groups = {}
    count = 0
    for gadget in schedule.gadgets:
        for index, fault in single_faults(circuit, gadget):
            count += 1
            outgoing, flipped = propagate_fault(circuit, gadget, index, fault)
            flag_flips = flipped & flags if gadget.half == "Z" else frozenset()
            if flag_flips:
                assert outgoing.weight() <= 2
                key = (gadget, flag_flips, flipped - flags - relays)
                groups.setdefault(key, []).append(outgoing)
            else:
                assert any(D5.code.in_gauge_group(outgoing * p) for p in candidates)
    assert count > 2000
    for errors in groups.values():
        for first, second in combinations(errors, 2):
            assert not D5.code.is_logical(first * second)
