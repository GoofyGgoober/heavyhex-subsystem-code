"""The local reproduction driver cannot overwrite the archived experiment."""

import numpy as np
import pytest

from followup import reproduce as r


def test_matched_rates_hold_transverse_decay():
    r1, rq = r.matched_rates(0.05, 3, 2, 0.01)
    assert rq == 0.01
    assert 3 * r1 + 2 * rq == pytest.approx(0.05)
    assert rq >= r1 / 2


def test_matched_rates_respect_physical_boundary():
    r1, rq = r.matched_rates(0.2, 3, 2, 0.01)
    assert rq == r1 / 2
    assert 3 * r1 + 2 * rq == pytest.approx(0.2)


@pytest.mark.parametrize("target,tz,te,rq", [(0.01, 3, 2, 0.01), (0.1, 0, 2, 0.01)])
def test_unmatchable_or_invalid_rates_rejected(target, tz, te, rq):
    with pytest.raises(ValueError):
        r.matched_rates(target, tz, te, rq)


def test_output_cannot_target_archive():
    with pytest.raises(ValueError):
        r.output_path(r.RUNS / "ibm_kingston-followup" / "analysis" / "life.json")


def test_output_does_not_overwrite(monkeypatch, tmp_path):
    monkeypatch.setattr(r, "REPO", tmp_path)
    build = tmp_path / "build"
    build.mkdir()
    used = build / "test.json"
    used.touch()
    with pytest.raises(FileExistsError):
        r.output_path(used)
    assert r.output_path(build / "new.json") == build / "new.json"


def test_strict_json_converts_nonfinite_values():
    assert r.clean({3: [np.float64(2), float("inf"), np.nan]}) == {"3": [2.0, None, None]}


@pytest.mark.parametrize("r1,phi", [(0.005, 0.003), (0.01, -0.004)])
def test_unconstrained_fit_recovers_rates_without_clipping(r1, phi):
    n = np.arange(1, 6, dtype=float)
    rq = r1 / 2 + phi
    tx, tz = 2.2 * n, 3.1 * n
    transverse = 0.99 * np.exp(-tz * r1 - tx * rq)
    longitudinal = 0.99 * np.exp(-tx * r1 - tz * rq)
    xy = {
        "x": transverse,
        "y": np.zeros(5),
        "shots_x": np.full(5, 1e12),
        "shots_y": np.full(5, 1e12),
        "equator_us": tx,
        "along_z_us": tz,
        "pulses": np.ones(5),
    }
    z = {
        "p0": (1 - longitudinal) / 2,
        "p1": (1 + longitudinal) / 2,
        "shots_0": np.full(5, 1e12),
        "shots_1": np.full(5, 1e12),
        "equator_us": tz,
        "along_z_us": tx,
        "pulses": np.ones(5),
    }
    fitted = r.unconstrained(xy, z)
    assert fitted["rate_t1"] == pytest.approx(r1, abs=1e-8)
    assert fitted["rate_t_phi"] == pytest.approx(phi, abs=1e-8)
