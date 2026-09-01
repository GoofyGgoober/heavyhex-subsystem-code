from surface_code import Pauli
from surface_code.patches import PATCH
from surface_code.simulation.cli import (
    MENU,
    format_pauli,
    format_report,
    format_syndrome,
    main,
    parse_error,
)


def test_parse_error_tokens():
    assert parse_error("") == Pauli()
    assert parse_error("I") == Pauli()
    assert parse_error("X0") == Pauli.x_on((0,))
    assert parse_error("Y4") == Pauli.x_on((4,)) * Pauli.z_on((4,))
    assert parse_error("X0 X3") == Pauli.x_on((0, 3))
    assert parse_error("X0Z3") == Pauli.x_on((0,)) * Pauli.z_on((3,))


def test_menu_lists_physical_logical_and_measure():
    assert MENU == (
        ("1", "Physical X_i"),
        ("2", "Physical Z_i"),
        ("3", "Logical X"),
        ("4", "Logical Z"),
        ("5", "Measure"),
        ("r", "Reset"),
    )


def test_format_pauli_and_syndrome_are_readable():
    assert format_pauli(Pauli()) == "I"
    assert format_pauli(Pauli.x_on((4,))) == "X4"
    assert format_pauli(PATCH.logical_x) == "X0 X3 X6  (X_L)"
    assert "trivial" in format_syndrome((0,) * 8)
    assert "Z-check 0" in format_syndrome((0, 0, 0, 0, 1, 1, 0, 0))


def test_format_report_says_whether_logical_survived():
    zeros = (0,) * 8
    report = format_report({(zeros, (0,) * 9): 7, (zeros, (0, 0, 0, 1, 0, 0, 0, 0, 0)): 3}, Pauli())
    assert "Error:       I" in report
    assert "Logical Z:   +1" in report
    assert "10 shots" in report
    assert "Syndromes:" not in report


def test_quit_from_menu():
    assert main(lines=["q"]) == 0


def test_measure_and_physical_x(monkeypatch, capsys):
    seen: list[Pauli] = []

    def fake_run(error, shots):
        seen.append(error)
        return {((0,) * 8, (0,) * 9): shots}

    monkeypatch.setattr("surface_code.simulation.cli.run_aer", fake_run)
    assert main(lines=["5", "1", "4", "q"]) == 0
    assert seen == [Pauli(), Pauli.x_on((4,))]
    assert "Logical Z:   +1" in capsys.readouterr().out


def test_z3_then_z3_returns_to_identity(monkeypatch):
    seen: list[Pauli] = []

    def fake_run(error, shots):
        seen.append(error)
        return {((0,) * 8, (0,) * 9): shots}

    monkeypatch.setattr("surface_code.simulation.cli.run_aer", fake_run)
    assert main(lines=["2", "3", "2", "3", "q"]) == 0
    assert seen == [Pauli.z_on((3,)), Pauli()]
