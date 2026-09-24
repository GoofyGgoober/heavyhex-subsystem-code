"""MWPM on hand-built syndrome histories. Corrections only need to be right up to a gauge."""

from itertools import combinations, product

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("pymatching")

from heavyhex.core import Pauli  # noqa: E402
from heavyhex.decoders.mwpm import MemoryNoise, MWPMDecoder  # noqa: E402
from heavyhex.decoders.shots import Memory, Shot, detectors_from_syndromes  # noqa: E402
from heavyhex.patches.operators import D3, D5  # noqa: E402
from heavyhex.simulation.phenomenological import benchmark_memory, sample_memory  # noqa: E402


def zero_syndromes(memory):
    return {
        (r, name): 0
        for r in range(-1, memory.rounds)
        for name in memory.patch.stabilizer_names
        if r >= 0 or not name.startswith(memory.basis)
    }


def error_shot(decoder, error, interval):
    """A shot with `error` on the data from round `interval` on, built without the graph."""
    memory = decoder.memory
    syndromes = zero_syndromes(memory)
    syndrome = dict(zip(memory.patch.stabilizer_names, memory.patch.code.syndrome(error)))
    for r, name in syndromes:
        if r >= interval:
            syndromes[r, name] = syndrome[name]
    flips = error.x if memory.basis == "Z" else error.z
    data = tuple(int(q in flips) for q in memory.patch.data_qubits)
    return Shot(memory, detectors_from_syndromes(memory, syndromes, data), data)


@pytest.mark.parametrize("patch", [D3, D5], ids=["d3", "d5"])
def test_every_correctable_data_error_is_corrected(patch):
    decoders = [MWPMDecoder(patch, basis=b, rounds=1) for b in ("X", "Z")]
    errors = [Pauli()]
    for weight in range(1, (patch.distance - 1) // 2 + 1):
        for support in combinations(patch.data_qubits, weight):
            for axes in product("XYZ", repeat=weight):
                errors.append(
                    Pauli(
                        frozenset(q for q, axis in zip(support, axes) if axis in "XY"),
                        frozenset(q for q, axis in zip(support, axes) if axis in "ZY"),
                    )
                )
    masks = [
        decoder.decode_batch([error_shot(decoder, e, 0).detectors for e in errors])
        for decoder in decoders
    ]
    for i, error in enumerate(errors):
        correction = Pauli(
            frozenset(int(q) for q in np.flatnonzero(masks[1][i])),
            frozenset(int(q) for q in np.flatnonzero(masks[0][i])),
        )
        assert patch.code.in_gauge_group(error * correction), error


@pytest.mark.parametrize("patch", [D3, D5], ids=["d3", "d5"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_every_data_location_and_time_interval_corrects(patch, basis):
    decoder = MWPMDecoder(patch, basis=basis, rounds=3)
    pauli = Pauli.x_on if basis == "Z" else Pauli.z_on
    for interval in range(4):
        for q in patch.data_qubits:
            shot = error_shot(decoder, pauli((q,)), interval)
            assert not decoder.fails(shot)
    # A logical error fires nothing, so it has to come out as a failure.
    logical = patch.code.logical_x if basis == "Z" else patch.code.logical_z
    shot = error_shot(decoder, logical, 0)
    assert not any(shot.detectors)
    assert decoder.fails(shot)


@pytest.mark.parametrize("patch", [D3, D5], ids=["d3", "d5"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_each_check_measurement_error_forms_a_time_pair(patch, basis):
    decoder = MWPMDecoder(patch, basis=basis, rounds=3)
    memory = decoder.memory
    data = (0,) * patch.code.n
    for r in range(3):
        for name in patch.stabilizer_names:
            if not name.startswith(basis):
                continue
            syndromes = zero_syndromes(memory)
            syndromes[r, name] = 1
            detectors = detectors_from_syndromes(memory, syndromes, data)
            assert {label for label, bit in zip(memory.labels, detectors) if bit} == {
                (r, name),
                (r + 1, name),
            }
            assert decoder.decode(detectors) == Pauli()
            assert not decoder.fails(Shot(memory, detectors, data))


def test_random_prep_values_are_the_reference():
    memory = Memory(D5, "Z", 3)
    syndromes = zero_syndromes(memory)
    for key in syndromes:
        if key[1].startswith("X"):
            syndromes[key] = 1
    detectors = detectors_from_syndromes(memory, syndromes, (0,) * 25)
    assert not any(detectors)
    assert all(name.startswith("Z") for r, name in memory.labels if r == 3)
    syndromes[-1, next(iter(D5.x_stabilizers))] = 0
    assert sum(detectors_from_syndromes(memory, syndromes, (0,) * 25)) == 1


def test_final_readout_is_not_assumed_perfect():
    decoder = MWPMDecoder(D5, basis="Z", rounds=3, noise=MemoryNoise(readout=0.02))
    shot = error_shot(decoder, Pauli.x_on((0,)), 3)
    assert all(r == 3 for (r, _), bit in zip(shot.memory.labels, shot.detectors) if bit)
    assert not decoder.fails(shot)


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_batch_matches_single_and_ignores_other_basis(basis):
    decoder = MWPMDecoder(D5, basis=basis, rounds=3)
    samples = sample_memory(D5, basis=basis, rounds=3, shots=32, seed=93)
    masks = decoder.decode_batch(samples.detectors)
    for detectors, mask in zip(samples.detectors, masks):
        correction = decoder.decode(tuple(int(bit) for bit in detectors))
        support = correction.x if basis == "Z" else correction.z
        assert support == frozenset(np.flatnonzero(mask))
    changed = samples.detectors.copy()
    others = sorted(set(range(len(decoder.memory.labels))) - set(decoder.memory.basis_indices))
    changed[:, others] ^= 1
    assert np.array_equal(masks, decoder.decode_batch(changed))
    assert decoder.decode_batch(
        np.empty((0, len(decoder.memory.labels)), dtype=np.uint8)
    ).shape == (0, 25)


@pytest.mark.parametrize("values", [[], [[0]], [[0.0] * 8], [[2] * 8], [[-1] * 8], [["0"] * 8]])
def test_decoder_rejects_invalid_batches(values):
    decoder = MWPMDecoder(D3)
    assert len(decoder.memory.labels) == 8
    with pytest.raises(ValueError):
        decoder.decode_batch(values)


@pytest.mark.parametrize("value", [True, -0.1, 0, 0.5, 1, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["data", "measurement"])
def test_invalid_noise_prior_rejected(value, field):
    with pytest.raises(ValueError):
        MemoryNoise(**{field: value})


def test_decoder_accepts_small_finite_probabilities():
    decoder = MWPMDecoder(D3, noise=MemoryNoise(data=1e-100, measurement=1e-100))
    assert decoder.decode((0,) * len(decoder.memory.labels)) == Pauli()


@pytest.mark.parametrize("rounds", [0, -1, True, 1.5])
def test_invalid_rounds_rejected(rounds):
    with pytest.raises(ValueError, match="positive integer"):
        MWPMDecoder(D3, rounds=rounds)


def test_incomplete_history_and_mismatched_shot_are_rejected():
    decoder = MWPMDecoder(D3)
    syndromes = zero_syndromes(decoder.memory)
    syndromes.pop(next(iter(syndromes)))
    with pytest.raises(ValueError, match="history"):
        detectors_from_syndromes(decoder.memory, syndromes, (0,) * 9)
    other = MWPMDecoder(D3, basis="X")
    with pytest.raises(ValueError, match="different distance"):
        decoder.fails(error_shot(other, Pauli(), 0))


@pytest.mark.parametrize("patch", [D3, D5], ids=["d3", "d5"])
@pytest.mark.parametrize("basis", ["X", "Z"])
def test_seeded_noisy_benchmark_improves_logical_error(patch, basis):
    result = benchmark_memory(patch, basis=basis, rounds=3, shots=4000, seed=29)
    assert result == benchmark_memory(patch, basis=basis, rounds=3, shots=4000, seed=29)
    assert 0 < result["logical_failures"] < result["raw_logical_flips"]


@pytest.mark.parametrize("basis", ["X", "Z"])
def test_all_pairs_of_d5_phenomenological_faults(basis):
    """Every pair of single faults: data flips in any round, readout flips, wrong outcomes."""
    decoder = MWPMDecoder(D5, basis=basis, rounds=3)
    memory = decoder.memory
    primitive_detectors, primitive_flips = [], []
    pauli = Pauli.x_on if basis == "Z" else Pauli.z_on
    for r in range(4):
        for q in D5.data_qubits:
            shot = error_shot(decoder, pauli((q,)), r)
            primitive_detectors.append(shot.detectors)
            primitive_flips.append(shot.logical_bit)
    for r in range(3):
        for name in D5.stabilizer_names:
            if name.startswith(basis):
                syndromes = zero_syndromes(memory)
                syndromes[r, name] = 1
                primitive_detectors.append(detectors_from_syndromes(memory, syndromes, (0,) * 25))
                primitive_flips.append(0)
    primitive_detectors = np.asarray(primitive_detectors, dtype=np.uint8)
    pairs = np.asarray(list(combinations(range(len(primitive_detectors)), 2)))
    detectors = primitive_detectors[pairs[:, 0]] ^ primitive_detectors[pairs[:, 1]]
    truth = np.asarray(primitive_flips, dtype=np.uint8)
    truth = truth[pairs[:, 0]] ^ truth[pairs[:, 1]]
    masks = decoder.decode_batch(detectors)
    predicted = masks[:, sorted(memory.logical_support)].sum(axis=1) % 2
    assert np.array_equal(predicted, truth)


def test_readout_noise_and_benchmark_interval():
    noise = MemoryNoise(data=0.002, measurement=0.003, readout=0.07)
    samples = sample_memory(D5, basis="X", rounds=2, shots=1000, seed=14, noise=noise)
    support = sorted(samples.memory.logical_support)
    assert np.array_equal(samples.logical_flips, samples.data_bits[:, support].sum(axis=1) % 2)
    result = benchmark_memory(D5, basis="X", rounds=2, shots=1000, seed=14, noise=noise)
    lower, upper = result["logical_error_rate_95ci"]
    assert 0 <= lower < result["logical_error_rate"] < upper <= 1
