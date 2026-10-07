"""Detectors read from Aer shots of the flagged circuit."""

import pytest

pytest.importorskip("qiskit_aer")

from heavyhex.circuits.flagged import memory_circuit_flagged  # noqa: E402
from heavyhex.circuits.shots import read_shot  # noqa: E402
from heavyhex.patches.operators import D3, D5  # noqa: E402
from heavyhex.simulation.aer import _sample_counts, _split_key, run_memory_flagged  # noqa: E402


@pytest.mark.parametrize("patch", [D3, D5], ids=["d3", "d5"])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 3])
def test_noiseless_circuit_fires_no_detectors(patch, basis, rounds):
    circuit, schedule = memory_circuit_flagged(patch, basis=basis, rounds=rounds)
    counts = _sample_counts(circuit, 64, 17)
    gauge_width = len(circuit.cregs[0])
    for key in counts:
        data, gauge = _split_key(key, patch.code.n, gauge_width)
        shot = read_shot(schedule, gauge, data)
        assert not any(shot.detectors)
        assert shot.logical_bit == 0
        last_row = sum(name.startswith(basis) for name in patch.stabilizer_names)
        assert len(shot.detectors) == rounds * patch.code.syndrome_size + last_row


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_logical_fault_is_invisible_and_flips_the_readout(basis):
    error = D5.code.logical_x if basis == "Z" else D5.code.logical_z
    records = run_memory_flagged(D5, rounds=3, basis=basis, error=error, shots=16, seed=3)
    assert all(not any(record["detectors"]) for record in records)
    assert all(record["logical_bit"] == 1 for record in records)


def test_invalid_gauge_and_data_records_are_rejected():
    _, schedule = memory_circuit_flagged(D5)
    gauge = (0,) * len(schedule.measurements)
    with pytest.raises(ValueError, match="gauge outcomes"):
        read_shot(schedule, gauge[:-1], (0,) * 25)
    with pytest.raises(ValueError, match="data readout"):
        read_shot(schedule, gauge, (0,) * 24)
    with pytest.raises(ValueError, match="gauge outcomes"):
        read_shot(schedule, (2,) + gauge[1:], (0,) * 25)
