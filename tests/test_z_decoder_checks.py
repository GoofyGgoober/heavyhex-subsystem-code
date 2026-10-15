"""Independent small-case checks of exact XOR probabilities and detector removal."""

import itertools

import numpy as np
import pytest
import stim


@pytest.fixture
def checks(monkeypatch, repo_root):
    monkeypatch.syspath_prepend(str(repo_root / "docs/analysis"))
    import z_decoder_checks

    return z_decoder_checks


@pytest.mark.parametrize("probabilities", [(0.1, 0.2, 0.3), (0.0, 0.5, 1.0), (0.8, 0.4, 0.5)])
def test_exact_xor_matches_exhaustive_fault_enumeration(checks, probabilities):
    masks = (1, 3, 6)
    expected = np.zeros(8)
    for pattern in itertools.product((0, 1), repeat=3):
        mask, weight = 0, 1
        for present, p, effect in zip(pattern, probabilities, masks, strict=True):
            weight *= p if present else 1 - p
            if present:
                mask ^= effect
        expected[mask] += weight
    actual = checks.xor_distribution(3, zip(masks, probabilities, strict=True))
    np.testing.assert_allclose(actual, expected, atol=1e-14)


def test_ml_marginalizes_other_patch_and_keeps_undetected_logical(checks):
    dem = stim.DetectorErrorModel("error(0.1) D0 L0\nerror(0.2) D1 L1\nerror(0.3) L0")
    table = checks.ml_table(dem, [0], 0)
    np.testing.assert_allclose(table, [[0.63, 0.27], [0.03, 0.07]])
    np.testing.assert_array_equal(checks.ml_decode(table, np.array([[0], [1]])), [False, True])


def test_reduction_preserves_measurements_observable_and_noise(checks):
    model = stim.Circuit(
        "R 0\nX_ERROR(0.2) 0\nM 0\nDETECTOR rec[-1]\nDETECTOR rec[-1]\nOBSERVABLE_INCLUDE(0) rec[-1]"
    )
    reduced, labels, keep = checks.stabilizers_only(model, [("3", 1, "Z1"), ("3", 1, "relay Z1")])
    assert labels == [("3", 1, "Z1")]
    assert keep.tolist() == [True, False]
    assert reduced.num_measurements == model.num_measurements
    assert reduced.num_observables == model.num_observables
    assert [i for i in reduced if i.name != "DETECTOR"] == [
        i for i in model if i.name != "DETECTOR"
    ]
    for measured in (False, True):
        raw = np.array([[measured]])
        d0, f0 = model.compile_m2d_converter().convert(measurements=raw, separate_observables=True)
        d1, f1 = reduced.compile_m2d_converter().convert(
            measurements=raw, separate_observables=True
        )
        np.testing.assert_array_equal(d1, d0[:, keep])
        np.testing.assert_array_equal(f1, f0)
