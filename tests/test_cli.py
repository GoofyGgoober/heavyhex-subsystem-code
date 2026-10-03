"""Smoke tests for the heavy-hex command-line tools."""

import json
import sys
from pathlib import Path

import pytest

from heavyhex.cli import COMMANDS, main, parse_pauli
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


def test_info_reports_patch_parameters(capsys):
    assert main(["--distance", "3", "info"]) == 0
    out = capsys.readouterr().out
    assert "[[9,1,3]]" in out and "X1X4" not in out
    assert main(["--distance", "5", "--json", "info"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["distance"] == 5 and payload["n_logical"] == 1


@pytest.mark.parametrize(
    "argv, message",
    [
        (["circuit", "--error", "Q1"], "cannot parse"),
        (["circuit", "--basis", "Y"], "invalid choice"),
    ],
)
def test_user_errors_exit_2_without_a_traceback(argv, message, capsys):
    assert main(argv) == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("argv", [[], ["nosuch"], ["experiment"], ["--distance", "4", "info"]])
def test_argparse_usage_errors_exit_2(argv):
    assert main(argv) == 2


@pytest.mark.parametrize("command", sorted(COMMANDS))
def test_every_command_has_help(command, capsys):
    assert main([command, "--help"]) == 0
    assert capsys.readouterr().out.startswith(f"usage: heavyhex {command}")


def test_missing_qiskit_extra_is_reported_not_raised(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "qiskit", None)
    assert main(["circuit"]) == 1
    assert "pip install" in capsys.readouterr().err


def test_d5_circuit_is_available(capsys):
    pytest.importorskip("qiskit")
    assert main(["--distance", "5", "circuit"]) == 0
    assert "q_64" in capsys.readouterr().out


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
