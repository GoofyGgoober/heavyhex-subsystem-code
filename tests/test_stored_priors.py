"""The stored forecast is compared with device counts decoded under its own priors."""

import hashlib
import json

import numpy as np
import pytest


@pytest.fixture
def stored(monkeypatch, repo_root):
    monkeypatch.syspath_prepend(str(repo_root / "docs" / "analysis"))
    import stored_priors

    return stored_priors


def test_cached_counts_reuse_only_matching_inputs(stored, monkeypatch, tmp_path):
    folder = tmp_path / "job"
    folder.mkdir()
    (folder / "analysis.json").write_text("{}")
    monkeypatch.setattr(stored, "CACHE", tmp_path / "cache")
    inputs = {"input": "first"}
    monkeypatch.setattr(stored, "fingerprint", lambda _: inputs.copy())
    calls = []

    def decode(_, f):
        calls.append(f)
        return {"X": {"3": [2, 4, 6, 8, 10, 12]}}

    monkeypatch.setattr(stored, "redecode", decode)
    first = stored.cached_counts(folder)
    assert first["decoder_f"] == 1.0
    assert stored.cached_counts(folder) == first
    assert calls == [1.0]
    inputs["input"] = "second"
    assert stored.cached_counts(folder)["inputs"] == inputs
    assert calls == [1.0, 1.0]


def test_primary_intervals_reconstruct_from_matched_counts(stored):
    saved = stored.read(stored.RESULTS / "stored_priors.json")
    fits = stored.read(stored.RESULTS / "fits.json")
    for quantity in stored.QUANTITIES:
        basis, name = quantity.split()
        daily = {}
        for rep in fits["per_rep"].values():
            eps = {d: saved["curves"][rep["jobs"][d]][basis][d]["binomial"]["eps"] for d in "35"}
            observed = eps["3"] / eps["5"] if name == "lambda" else eps[name[-1]]
            model = stored.predicted(fits["predictions"], rep, stored.F1, quantity)
            daily.setdefault(rep["day"], []).append(np.log(observed / model))
        y = np.array([np.mean(daily[day]) for day in sorted(daily)])
        half = 4.604094871349992 * y.std(ddof=1) / np.sqrt(5)
        result = saved["primary"][quantity]
        assert result["day_log_means"] == pytest.approx(y)
        assert result["ratio"] == pytest.approx(np.exp(y.mean()))
        assert result["ci95_family_of_five"] == pytest.approx(
            np.exp([y.mean() - half, y.mean() + half])
        )


def test_saved_counts_and_control_are_consistent(stored):
    saved = stored.read(stored.RESULTS / "stored_priors.json")
    original = stored.read(stored.RESULTS / "fits_counts.json")["jobs"]
    assert saved["adaptive_control_matches"] is True
    assert len(saved["curves"]) == 15
    above, total = 0, 0
    for job, bases in saved["curves"].items():
        for basis, patches in bases.items():
            for patch, curve in patches.items():
                assert curve["N"] == 2500
                assert curve["P"] == pytest.approx(np.array(curve["k"]) / 2500)
                predicted = original[job]["predicted"][stored.F1]["P"][basis][patch]
                above += sum(o > m for o, m in zip(curve["P"][:4], predicted[:4], strict=True))
                total += 4
    assert (saved["early_points_above"], saved["early_points_total"]) == (above, total)
    assert (above, total) == (160, 160)


def test_result_records_current_driver_fingerprint(stored):
    saved = stored.read(stored.RESULTS / "stored_priors.json")
    path = stored.Path(stored.__file__)
    key = str(path.relative_to(stored.REPO))
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    for inputs in saved["inputs"].values():
        assert inputs[key] == expected


def test_observed_prediction_figure_uses_matched_values(monkeypatch, repo_root, stored):
    from matplotlib import font_manager

    # This checks data selection, not rendering; it must not require a TeX font install.
    entry = font_manager.FontEntry(
        fname=font_manager.findfont("DejaVu Serif"), name="Latin Modern Roman"
    )
    monkeypatch.setattr(font_manager.fontManager, "ttflist", [entry])
    monkeypatch.syspath_prepend(str(repo_root / "paper" / "figures"))
    import fig_obs_pred

    rows = fig_obs_pred.curves(stored.RESULTS)
    assert len(rows) == 40
    assert any(r["obs"] != r["obs_measured"] for r in rows)
    assert all(r["lo"] <= r["obs"] <= r["hi"] for r in rows)
    raw = json.loads(fig_obs_pred.KINGSTON.read_text())
    king = fig_obs_pred.kingston()
    for row in king:
        expected = raw["observed_prerun_model_decoder"]["0.1"][row["basis"]]["eps"][row["d"]]
        assert row["obs"] == expected["binomial"][0]
