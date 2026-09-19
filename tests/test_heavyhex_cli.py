"""Smoke tests for the heavy-hex command-line tools."""

import json

import pytest

from surface_code.patches import D3
from surface_code.simulation.cli import main, parse_error, parse_syndrome

pytest.importorskip("qiskit_aer")


def test_parse_error_accepts_paulis_logicals_and_identity():
    from surface_code import Pauli

    assert parse_error("X0 Z3", D3) == Pauli(frozenset({0}), frozenset({3}))
    assert parse_error("I", D3).weight() == 0
    assert parse_error("X_L", D3) == D3.code.logical_x
    with pytest.raises(ValueError, match="data qubit"):
        parse_error("X9", D3)
    with pytest.raises(ValueError, match="cannot parse"):
        parse_error("Q1", D3)


def test_parse_syndrome_validates_length():
    assert parse_syndrome("000000", D3) == (0,) * 6
    with pytest.raises(ValueError, match="6 bits"):
        parse_syndrome("00000", D3)


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


def test_sweep_grades_weight_one(capsys):
    assert main(["sweep", "--weight", "1"]) == 0
    out = capsys.readouterr().out
    assert "Harmful:   0" in out
    assert main(["--json", "sweep", "--weight", "1"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["total"] == 27 and payload["harmful"] == 0


def test_run_reports_successes(capsys):
    assert main(["run", "--shots", "32", "--seed", "7", "--error", "X0"]) == 0
    out = capsys.readouterr().out
    assert "successes: 32" in out


def test_bad_input_returns_exit_code_2(capsys):
    assert main(["syndrome", "Q1"]) == 2
    assert main(["decode", "010"]) == 2
