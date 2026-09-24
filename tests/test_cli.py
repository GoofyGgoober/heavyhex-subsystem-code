"""Smoke tests for the heavy-hex command-line tools."""

import json
import sys
from pathlib import Path

import pytest

from heavyhex.cli import COMMANDS, main, parse_pauli, parse_syndrome
from heavyhex.core import Pauli
from heavyhex.patches.operators import D3


@pytest.mark.parametrize(
    "text, expected",
    [
        ("X0 Z3", Pauli(frozenset({0}), frozenset({3}))),
        ("y2", Pauli(frozenset({2}), frozenset({2}))),
        ("I", Pauli()),
        ("X_L", D3.code.logical_x),
        ("ZL", D3.code.logical_z),
    ],
)
def test_parse_pauli_accepts_paulis_logicals_and_identity(text, expected):
    assert parse_pauli(text, D3) == expected


@pytest.mark.parametrize(
    "text, message",
    [("X9", "data qubit"), ("Q1", "cannot parse"), ("", "expected a Pauli")],
)
def test_parse_pauli_rejects_bad_input(text, message):
    with pytest.raises(ValueError, match=message):
        parse_pauli(text, D3)


@pytest.mark.parametrize("text", ["00000", "00000x"])
def test_parse_syndrome_validates_width_and_alphabet(text):
    assert parse_syndrome("000000", D3) == (0,) * 6
    with pytest.raises(ValueError, match="6 bits"):
        parse_syndrome(text, D3)


def test_info_reports_patch_parameters(capsys):
    assert main(["--distance", "3", "info"]) == 0
    out = capsys.readouterr().out
    assert "[[9,1,3]]" in out and "X1X4" not in out
    assert main(["--distance", "5", "--json", "info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["distance"] == 5 and payload["n_logical"] == 1


def test_syndrome_and_decode_roundtrip(capsys):
    assert main(["syndrome", "X0"]) == 0
    assert "Correction" in capsys.readouterr().out
    assert main(["decode", "000011"]) == 0
    out = capsys.readouterr().out
    assert "Correction" in out and "Weight" in out
    assert main(["--distance", "5", "decode", "0" * 16]) == 2


def test_syndrome_at_d5_reports_detection_only(capsys):
    assert main(["--distance", "5", "--json", "syndrome", "X0"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert "correction" not in payload and payload["syndrome"].count("1") == 1


def test_sweep_grades_weight_one(capsys):
    assert main(["sweep", "--weight", "1"]) == 0
    out = capsys.readouterr().out
    assert "Harmful:   0" in out
    assert main(["--json", "sweep", "--weight", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 27 and payload["harmful"] == 0


def test_run_reports_successes(capsys):
    pytest.importorskip("qiskit_aer")
    assert main(["run", "--shots", "32", "--seed", "7", "--error", "X0"]) == 0
    assert "successes: 32" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv, message",
    [
        (["syndrome", "Q1"], "cannot parse"),
        (["decode", "010"], "6 bits"),
        (["run", "--basis", "Y"], "invalid choice"),
        (["sweep", "--weight", "1", "--axes", "Q"], "selection from XYZ"),
        (["sweep", "--weight", "99"], "weight must be in 0..9"),
        (["sweep", "--weight", "-1"], "weight must be in 0..9"),
    ],
)
def test_user_errors_exit_2_without_a_traceback(argv, message, capsys):
    assert main(argv) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["nosuch"], ["sweep"], ["--distance", "4", "info"]])
def test_argparse_usage_errors_exit_2(argv):
    assert main(argv) == 2


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_every_command_has_help(command, capsys):
    assert main([command, "--help"]) == 0
    assert capsys.readouterr().out.startswith(f"usage: heavyhex {command}")


@pytest.mark.parametrize("argv", [["run", "--shots", "4"], ["circuit"]])
def test_missing_qiskit_extra_is_reported_not_raised(argv, monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "qiskit", None)
    monkeypatch.setitem(sys.modules, "qiskit_aer", None)
    assert main(argv) == 1
    assert "pip install" in capsys.readouterr().err


def test_d5_run_is_explicitly_ungraded(capsys):
    pytest.importorskip("qiskit_aer")
    assert main(["--distance", "5", "--json", "run", "--shots", "8", "--seed", "7"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shots"] == 8
    assert payload["successes"] is None
    assert payload["dominant_syndrome"] == "0" * 16


def test_d5_circuit_is_available(capsys):
    pytest.importorskip("qiskit")
    assert main(["--distance", "5", "circuit"]) == 0
    assert "q_64" in capsys.readouterr().out


def test_d5_mwpm_run_decodes_late_error(capsys):
    pytest.importorskip("qiskit_aer")
    pytest.importorskip("pymatching")
    assert (
        main(
            [
                "--distance",
                "5",
                "--json",
                "run",
                "--decoder",
                "mwpm",
                "--rounds",
                "3",
                "--shots",
                "8",
                "--seed",
                "7",
                "--error",
                "X0",
                "--inject-at",
                "after_x2",
            ]
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["successes"] == 8
    assert payload["rounds"] == 3
    assert payload["decoder"] == "mwpm"


def test_phenomenological_simulation_cli(capsys):
    pytest.importorskip("pymatching")
    assert main(["--distance", "5", "--json", "mwpm-sim", "--shots", "128", "--seed", "7"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shots"] == 128
    assert payload["model"] == "phenomenological_independent_stabilizer_errors"
    assert payload["logical_failures"] <= payload["raw_logical_flips"]


def test_missing_matching_extra_is_reported(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "pymatching", None)
    assert main(["mwpm-sim", "--shots", "1"]) == 1
    assert "[sim,matching]" in capsys.readouterr().err


def test_calibrate_offline_reports_the_saved_picks(capsys):
    figures = Path(__file__).resolve().parents[1] / "docs" / "figures"
    argv = ["--json", "calibrate", "--offline", "--file", str(figures / "fez_calibration.json")]
    assert main(argv + ["--map", str(figures / "fez_map.json")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert set(payload["spots"]) == {"3", "5"}
    assert payload["spots"]["3"]["broken"] == []


def test_missing_hardware_extra_is_reported(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "qiskit_ibm_runtime", None)
    assert main(["calibrate", "--file", "/nonexistent/calibration.json"]) == 1
    assert "[hardware]" in capsys.readouterr().err
