"""Detectors from Aer shots of both circuits, decoded with MWPM."""

import pytest

pytest.importorskip("qiskit_aer")
pytest.importorskip("pymatching")

from heavyhex.circuits.flagged import memory_circuit_flagged  # noqa: E402
from heavyhex.circuits.ideal import memory_circuit  # noqa: E402
from heavyhex.core import Pauli  # noqa: E402
from heavyhex.decoders.mwpm import MWPMDecoder  # noqa: E402
from heavyhex.decoders.shots import read_shot  # noqa: E402
from heavyhex.patches.operators import D3, D5  # noqa: E402
from heavyhex.simulation.aer import _sample_counts, _split_key, run_memory_flagged  # noqa: E402


@pytest.mark.parametrize("patch", [D3, D5], ids=["d3", "d5"])
@pytest.mark.parametrize("basis", ["X", "Z"])
@pytest.mark.parametrize("rounds", [1, 3])
@pytest.mark.parametrize(
    "builder", [memory_circuit, memory_circuit_flagged], ids=["ideal", "flagged"]
)
def test_noiseless_circuit_fires_no_detectors(patch, basis, rounds, builder):
    circuit, schedule = builder(patch, basis=basis, rounds=rounds)
    counts = _sample_counts(circuit, 64, 17)
    gauge_width = len(circuit.cregs[0])
    decoder = MWPMDecoder(patch, basis=basis, rounds=rounds)
    for key in counts:
        data, gauge = _split_key(key, patch.code.n, gauge_width)
        shot = read_shot(schedule, gauge, data)
        assert not any(shot.detectors)
        assert shot.logical_bit == 0
        assert not decoder.fails(shot)
        last_row = sum(name.startswith(basis) for name in patch.stabilizer_names)
        assert len(shot.detectors) == rounds * patch.code.syndrome_size + last_row


@pytest.mark.parametrize("patch", [D3, D5], ids=["d3", "d5"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_every_data_pauli_between_any_halves_is_corrected(patch, basis):
    stages = ["after_prep"] + [f"after_{half}{r}" for r in range(3) for half in ("x", "z")]
    for q in patch.data_qubits:
        x, z = Pauli.x_on((q,)), Pauli.z_on((q,))
        for error in (x, z, x * z):
            for stage in stages:
                records = run_memory_flagged(
                    patch,
                    basis=basis,
                    rounds=3,
                    error=error,
                    inject_at=stage,
                    shots=4,
                    seed=19,
                    decoder="mwpm",
                )
                assert all(record["success"] for record in records), (q, error, stage)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_logical_fault_is_not_mistaken_for_success(basis):
    error = D5.code.logical_x if basis == "Z" else D5.code.logical_z
    records = run_memory_flagged(
        D5, rounds=3, basis=basis, error=error, shots=16, seed=3, decoder="mwpm"
    )
    assert all(not any(record["detectors"]) for record in records)
    assert all(record["success"] is False for record in records)


def test_invalid_gauge_and_data_records_are_rejected():
    _, schedule = memory_circuit_flagged(D5)
    gauge = (0,) * len(schedule.measurements)
    with pytest.raises(ValueError, match="gauge outcomes"):
        read_shot(schedule, gauge[:-1], (0,) * 25)
    with pytest.raises(ValueError, match="data readout"):
        read_shot(schedule, gauge, (0,) * 24)
    with pytest.raises(ValueError, match="gauge outcomes"):
        read_shot(schedule, (2,) + gauge[1:], (0,) * 25)


def test_unknown_decoder_rejected():
    with pytest.raises(ValueError, match="decoder"):
        run_memory_flagged(D5, decoder="unknown", shots=1)
