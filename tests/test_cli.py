"""Smoke tests for the heavy-hex command-line tools."""

import json
import sys

import pytest

from heavyhex.cli import COMMANDS, main, parse_error, parse_syndrome
from heavyhex.core import Pauli
from heavyhex.patches import D3


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
def test_parse_error_accepts_paulis_logicals_and_identity(text, expected):
    assert parse_error(text, D3) == expected


@pytest.mark.parametrize(
    "text, message",
    [("X9", "data qubit"), ("Q1", "cannot parse"), ("", "expected a Pauli")],
)
def test_parse_error_rejects_bad_input(text, message):
    with pytest.raises(ValueError, match=message):
        parse_error(text, D3)


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
        (["--distance", "5", "run"], "d=3 only"),
        (["--distance", "5", "circuit"], "d=3 only"),
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
