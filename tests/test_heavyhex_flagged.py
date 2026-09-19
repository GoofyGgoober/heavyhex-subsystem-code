"""Flagged d3 circuits: structure, deflagging, and single-fault tolerance."""

import json
from pathlib import Path

import pytest

from surface_code.circuits.heavyhex_flagged import (
    FALCON,
    NUM_QUBITS,
    memory_circuit_flagged,
    single_faults,
    with_fault,
)
from surface_code.core import Pauli
from surface_code.patches.heavyhex import D3

pytest.importorskip("qiskit_aer")

from surface_code.simulation.heavyhex_aer import (  # noqa: E402
    run_flagged_circuit,
    run_memory_flagged,
)

CODE = D3.code
FIGURES = Path(__file__).resolve().parents[1] / "docs" / "figures"


def test_circuit_uses_23_qubits_and_18_measurements_per_round():
    circuit, schedule = memory_circuit_flagged(rounds=1, basis="Z")
    assert circuit.num_qubits == NUM_QUBITS == 23
    # Prep X half (6) + Z half (12) + X half (6).
    assert len(schedule.measurements) == 24
    kinds = [m.kind for m in schedule.measurements if m.round == 0]
    assert sorted(kinds).count("z_syn") == 4
    assert sorted(kinds).count("z_flag") == 4
    assert sorted(kinds).count("relay") == 4
    assert sorted(kinds).count("x_gauge") == 6
    assert len(schedule.fragments) == 6 + 10


def test_falcon_roles_cover_the_documented_23_qubit_patch():
    assert len(set(FALCON.values())) == NUM_QUBITS
    device_map = json.loads((FIGURES / "fez_map.json").read_text())
    assert max(FALCON.values()) < len(device_map["coords"])


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_noiseless_flagged_memory_survives(basis):
    records = run_memory_flagged(rounds=1, basis=basis, shots=200, seed=7)
    assert all(record["success"] for record in records)


def test_relays_return_to_zero_in_the_absence_of_faults():
    _, schedule = memory_circuit_flagged(rounds=1, basis="Z")
    relay_bits = [m.bit for m in schedule.measurements if m.kind == "relay"]
    assert len(relay_bits) == 4
    records = run_memory_flagged(rounds=1, basis="Z", shots=100, seed=9)
    for record in records:
        assert all(record["gauge_bits"][bit] == 0 for bit in relay_bits)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_every_single_qubit_data_error_is_corrected_flagged(basis):
    for qubit in CODE.data_qubits:
        for error in (
            Pauli.x_on((qubit,)),
            Pauli.z_on((qubit,)),
            Pauli.x_on((qubit,)) * Pauli.z_on((qubit,)),
        ):
            records = run_memory_flagged(basis=basis, error=error, shots=16, seed=11)
            assert all(record["syndrome"] == CODE.syndrome(error) for record in records)
            assert all(record["success"] for record in records), (basis, error)


def test_lone_flags_are_deflagged_to_success():
    _, schedule = memory_circuit_flagged(rounds=1, basis="Z")
    flag_bits = {(m.round, m.gauge): m.bit for m in schedule.measurements if m.kind == "z_flag"}
    records = run_memory_flagged(rounds=1, basis="Z", shots=400, seed=13)
    lone = [
        record
        for record in records
        if (record["gauge_bits"][flag_bits[0, "X2X5"]] == 1)
        != (record["gauge_bits"][flag_bits[0, "X3X6"]] == 1)
    ]
    assert len(lone) > 50  # lone-flag branches are common, not corner cases
    assert all(record["success"] for record in lone)


def _in_gauge_times_weight_one(pauli: Pauli) -> bool:
    candidates = [Pauli()]
    for qubit in CODE.data_qubits:
        candidates.extend(
            [
                Pauli.x_on((qubit,)),
                Pauli.z_on((qubit,)),
                Pauli.x_on((qubit,)) * Pauli.z_on((qubit,)),
            ]
        )
    return any(CODE.in_gauge_group(pauli * candidate) for candidate in candidates)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_single_faults_never_grow_or_go_silent(basis):
    """Per gadget-local single fault, by Pauli propagation + sampled detectors:

    1. Unflagged faults leave data errors gauge-equivalent to weight <= 1.
       Flagged faults (a Z-half flag outcome flips) leave weight <= 2, and
       faults sharing a gadget + flag/gauge flip signature leave errors that
       differ detectably or by a gauge operator (never by a bare logical).
    2. All shots share one syndrome (detectors are deterministic given a fault).
    3. When no flag outcome flips and the sampled syndrome equals the outgoing
       error's algebraic syndrome (fully visible, exact deflag), every shot is
       corrected. Anything else is a measurement-like or after-last-check
       fault that needs multi-round matching, which must not be silently
       graded here.
    """
    from itertools import combinations

    from surface_code.circuits.heavyhex_flagged import FLAG_OF_X, propagate_fault

    flag_qubits = set(FLAG_OF_X.values())
    relay_qubits = {19, 20, 21, 22}
    circuit, schedule = memory_circuit_flagged(rounds=1, basis=basis)
    faults = [
        (fragment, index, fault)
        for fragment in schedule.fragments
        for index, fault in single_faults(circuit, fragment.start, fragment.end)
    ]
    assert len(faults) > 500
    flagged_groups: dict[tuple, list[Pauli]] = {}
    applied = asserted_nontrivial = 0
    for fragment, index, fault in faults:
        outgoing, flipped = propagate_fault(circuit, fragment.start, fragment.end, index, fault)
        flag_flips = flipped & flag_qubits if fragment.half == "Z" else frozenset()
        if not flag_flips:
            assert _in_gauge_times_weight_one(outgoing), (
                basis,
                fragment.name,
                index,
                fault,
            )
        else:
            assert outgoing.weight() <= 2, (basis, fragment.name, index, fault)
            gauge_flips = frozenset(flipped - flag_qubits - relay_qubits)
            key = (fragment.name, flag_flips, gauge_flips)
            flagged_groups.setdefault(key, []).append(outgoing)
        faulty = with_fault(circuit, index, fault)
        records = run_flagged_circuit(faulty, schedule, shots=8, seed=17)
        syndromes = {record["syndrome"] for record in records}
        assert len(syndromes) == 1, (basis, fragment.name, index, fault)
        syndrome = syndromes.pop()
        if not flag_flips and syndrome == CODE.syndrome(outgoing):
            applied += 1
            asserted_nontrivial += any(syndrome)
            assert all(record["success"] for record in records), (
                basis,
                fragment.name,
                index,
                fault,
            )
    assert applied > 100
    assert asserted_nontrivial > 50
    assert len(flagged_groups) > 10
    for key, errors in flagged_groups.items():
        for first, second in combinations(errors, 2):
            assert not CODE.is_harmful_undetectable(first * second), (basis, key)


def test_flagged_memory_rejects_bad_options():
    from surface_code.patches.heavyhex import build_operators

    with pytest.raises(ValueError, match="positive integer"):
        run_memory_flagged(shots=0)
    with pytest.raises(ValueError, match="d=3 only"):
        memory_circuit_flagged(build_operators(5))
