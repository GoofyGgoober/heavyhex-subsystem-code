"""Aer validation of abstract heavy-hex memory: stability, randomness, correction."""

import pytest

from surface_code.core import Pauli
from surface_code.patches.heavyhex import D3

pytest.importorskip("qiskit_aer")

from surface_code.simulation.heavyhex_aer import run_memory  # noqa: E402

CODE = D3.code


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_noiseless_round_has_trivial_syndrome_and_survives(basis):
    records = run_memory(rounds=1, basis=basis, shots=64, seed=7)
    assert len(records) == 64
    assert all(record["syndrome"] == (0,) * 6 for record in records)
    assert all(record["success"] for record in records)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_every_single_qubit_data_error_is_corrected(basis):
    for qubit in CODE.data_qubits:
        for error in (
            Pauli.x_on((qubit,)),
            Pauli.z_on((qubit,)),
            Pauli.x_on((qubit,)) * Pauli.z_on((qubit,)),
        ):
            records = run_memory(basis=basis, error=error, shots=16, seed=11)
            assert all(record["syndrome"] == CODE.syndrome(error) for record in records)
            assert all(record["success"] for record in records), (basis, error)


@pytest.mark.parametrize("basis", ["Z", "X"])
def test_logical_error_is_invisible_and_fatal(basis):
    error = CODE.logical_x if basis == "Z" else CODE.logical_z
    records = run_memory(basis=basis, error=error, shots=8, seed=3)
    assert all(record["syndrome"] == (0,) * 6 for record in records)
    assert not any(record["success"] for record in records)


def test_checks_are_frozen_across_noiseless_rounds_while_gauges_jitter():
    from surface_code.circuits.heavyhex import memory_circuit

    _, schedule = memory_circuit(D3, rounds=3, basis="Z")
    round_bits = [[slot.bit for slot in schedule.gauges if slot.round == r] for r in range(3)]
    records = run_memory(rounds=3, basis="Z", shots=32, seed=5)
    stabs = D3.stabilizer_names
    for record in records:
        checks = record["checks"]
        for name in stabs:
            assert checks[0, name] == checks[1, name] == checks[2, name]
        snapshots = {tuple(record["gauge_bits"][bit] for bit in bits) for bits in round_bits}
        assert len(snapshots) > 1


def test_memory_rejects_bad_options():
    with pytest.raises(ValueError, match="positive integer"):
        run_memory(shots=0)
    with pytest.raises(ValueError, match="X or Z"):
        run_memory(basis="Y")
