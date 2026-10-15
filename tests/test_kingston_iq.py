"""Posterior probabilities checked by enumerating all hidden-state paths."""

import itertools

import numpy as np
import pytest


@pytest.fixture
def iq(monkeypatch, repo_root):
    monkeypatch.syspath_prepend(str(repo_root / "docs/analysis"))
    import kingston_iq

    return kingston_iq


def test_posterior_against_path_enumeration(iq):
    fit = {"pi": 0.2, "a": 0.1, "b": 0.3, "e_N": [0.8, 0.15, 0.05], "e_P": [0.1, 0.7, 0.2]}
    observations = np.array([[0, 1, 2], [1, 0, 1]])
    transition = np.array([[0.9, 0.1], [0.3, 0.7]])
    emission = np.array([fit["e_N"], fit["e_P"]])
    expected = []
    for observed in observations:
        total, numerator = 0.0, np.zeros(3)
        for states in itertools.product((0, 1), repeat=3):
            weight = [0.8, 0.2][states[0]] * emission[states[0], observed[0]]
            for t in (1, 2):
                weight *= transition[states[t - 1], states[t]] * emission[states[t], observed[t]]
            total += weight
            numerator += weight * np.array(states)
        expected.append(numerator / total)
    np.testing.assert_allclose(iq.posterior(observations, fit), expected)


def test_projection_respects_rotated_reference_axis(iq):
    cloud = iq.lc.Cloud
    clouds = iq.lc.Clouds(cloud(np.array([2, 3]), np.eye(2)), cloud(np.array([2, 5]), np.eye(2)))
    along, across = iq.project(np.array([2 + 3j, 2 + 5j, 0 + 4j]), clouds)
    np.testing.assert_allclose(along, [0, 1, 0.5])
    np.testing.assert_allclose(across, [0, 0, 1])


def test_pair_correlations_are_occupancy_not_onset(iq):
    x = np.array([[0.1, 0.2, 0.8], [0.7, 0.3, 0.1]])
    rows = iq.pair_correlations({1: x, 2: x, 3: 1 - x}, {1: [5], 2: [5], 3: [6]})
    assert rows[0]["shared_partners"] == [5]
    assert rows[0]["posterior_correlation"] == pytest.approx(1)
    assert rows[1]["posterior_correlation"] == pytest.approx(-1)
