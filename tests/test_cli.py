import json
import subprocess
import sys

import pytest

from surface_code import Pauli
from surface_code.patches import PATCH
from surface_code.simulation.cli import (
    MENU,
    format_info,
    format_pauli,
    format_report,
    format_syndrome,
    main,
    parse_error,
    parse_syndrome,
)


ZERO_SYNDROME = (0,) * 8
ZERO_DATA = (0,) * 9


def _fake_tallies(shots: int):
    return {(ZERO_SYNDROME, ZERO_DATA): shots}


def test_parse_error_supports_physical_logical_and_separators():
    assert parse_error("") == Pauli()
    assert parse_error("I") == Pauli()
    assert parse_error("x0") == Pauli.x_on((0,))
    assert parse_error("Y4") == Pauli.x_on((4,)) * Pauli.z_on((4,))
    assert parse_error("X0, Z3") == Pauli.x_on((0,)) * Pauli.z_on((3,))
    assert parse_error("XL") == PATCH.logical_x
    assert parse_error("logical-z") == PATCH.logical_z
    assert parse_error("YL") == PATCH.logical_x * PATCH.logical_z


def test_parse_error_composes_repeated_tokens():
    assert parse_error("X0 X0") == Pauli()
    assert parse_error("X0 Z0") == Pauli.x_on((0,)) * Pauli.z_on((0,))
    assert parse_error("Y0 X0") == Pauli.z_on((0,))


@pytest.mark.parametrize("text", ["X9", "Z-1", "A0", "X", "X0 garbage"])
def test_parse_error_rejects_invalid_expressions(text):
    with pytest.raises(ValueError):
        parse_error(text)


def test_parse_and_format_syndrome():
    syndrome = (0, 0, 0, 0, 1, 1, 0, 0)
    assert parse_syndrome("0000 1100") == syndrome
    assert "Z-check 0" in format_syndrome(syndrome)
    assert format_syndrome(ZERO_SYNDROME) == "0000 0000  (trivial)"
    with pytest.raises(ValueError):
        parse_syndrome("000")
    with pytest.raises(ValueError):
        format_syndrome((0, 2) + (0,) * 6)


def test_menu_exposes_edit_measure_and_inspection_commands():
    labels = " ".join(key + " " + label for key, label in MENU).lower()
    for expected in ("physical x", "physical y", "logical x", "measure", "undo", "circuit", "info"):
        assert expected in labels


def test_format_pauli_and_info_are_readable():
    assert format_pauli(Pauli()) == "I"
    assert format_pauli(Pauli.x_on((4,))) == "X4"
    assert format_pauli(PATCH.logical_x) == "X0 X3 X6  (X_L)"
    info = format_info()
    assert "[[9,1,3]]" in info
    assert "0  1  2" in info
    assert "ancilla 9" in info


def test_format_report_summarizes_and_can_show_counts():
    tallies = {
        (ZERO_SYNDROME, ZERO_DATA): 7,
        (ZERO_SYNDROME, (0, 0, 0, 1, 0, 0, 0, 0, 0)): 3,
    }
    report = format_report(tallies, Pauli(), show_counts=True)
    assert "Aer stabilizer (ideal circuit)" in report
    assert "Error:        I" in report
    assert "Decoded Z_L:  +1" in report
    assert "10 shots" in report
    assert "syndrome  data" in report


def test_format_report_rejects_empty_or_malformed_results():
    with pytest.raises(ValueError, match="no outcomes"):
        format_report({}, Pauli())
    with pytest.raises(ValueError, match="positive integer"):
        format_report({(ZERO_SYNDROME, ZERO_DATA): 0}, Pauli())
    with pytest.raises(ValueError, match="data result"):
        format_report({(ZERO_SYNDROME, (0,)): 1}, Pauli())


def test_format_report_breaks_syndrome_ties_deterministically():
    other = (1,) + (0,) * 7
    items = [((other, ZERO_DATA), 1), ((ZERO_SYNDROME, ZERO_DATA), 1)]
    assert format_report(dict(items), Pauli()) == format_report(dict(reversed(items)), Pauli())


def test_help_version_and_bad_options_have_standard_exit_codes(capsys):
    assert main(["--help"]) == 0
    assert "COMMAND" in capsys.readouterr().out
    assert main(["--version"]) == 0
    assert "0.1.0" in capsys.readouterr().out
    assert main(["--definitely-not-an-option"]) == 2
    assert "unrecognized arguments" in capsys.readouterr().err


@pytest.mark.parametrize("shots", ["0", "-1", "nope"])
def test_run_rejects_invalid_shot_counts(shots, capsys):
    assert main(["run", "--shots", shots]) == 2
    assert "positive integer" in capsys.readouterr().err


def test_run_rejects_seed_above_aer_limit(capsys):
    assert main(["run", "--seed", str(1 << 63)]) == 2
    assert "seed must be at most" in capsys.readouterr().err


def test_batch_run_passes_error_shots_and_seed(monkeypatch, capsys):
    seen = []

    def fake_run(error, *, shots, seed):
        seen.append((error, shots, seed))
        return _fake_tallies(shots)

    monkeypatch.setattr("surface_code.simulation.cli.run_aer", fake_run)
    assert main(["X4", "--shots", "12", "--seed", "7", "--counts"]) == 0
    assert seen == [(Pauli.x_on((4,)), 12, 7)]
    output = capsys.readouterr().out
    assert "Error:        X4" in output
    assert "syndrome  data" in output


def test_batch_json_is_machine_readable(monkeypatch, capsys):
    monkeypatch.setattr(
        "surface_code.simulation.cli.run_aer",
        lambda error, *, shots, seed: _fake_tallies(shots),
    )
    assert main(["run", "I", "--shots", "3", "--seed", "2", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["shots"] == 3
    assert payload["error"] == "I"
    assert payload["circuit_noise"] == "none"
    assert "outcomes" not in payload


def test_nonzero_random_error_defaults_to_one_reproducible_draw(monkeypatch, capsys):
    draws = []
    seen = []

    def fake_noise(p, qubits, rng):
        draws.append((p, tuple(qubits), rng.random()))
        return Pauli.z_on((2,))

    def fake_run(error, *, shots, seed):
        seen.append((error, shots, seed))
        return _fake_tallies(shots)

    monkeypatch.setattr("surface_code.simulation.cli.depolarizing_error", fake_noise)
    monkeypatch.setattr("surface_code.simulation.cli.run_aer", fake_run)
    assert main(["run", "--draw-error", "0.2", "--seed", "7", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["random_error_draw"] == {
        "draws": 1,
        "probability": 0.2,
        "sampled_error": "Z2",
    }
    assert seen == [(Pauli.z_on((2,)), 64, 7)]
    assert draws[0][:2] == (0.2, PATCH.data_qubits)


def test_explicit_zero_probability_still_records_one_draw(monkeypatch, capsys):
    monkeypatch.setattr(
        "surface_code.simulation.cli.run_aer",
        lambda error, shots: _fake_tallies(shots),
    )
    assert main(["run", "--draw-error", "0", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["random_error_draw"] == {
        "draws": 1,
        "probability": 0.0,
        "sampled_error": "I",
    }


def test_missing_qiskit_message_is_narrow(monkeypatch, capsys):
    def missing_aer(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'qiskit_aer'", name="qiskit_aer")

    monkeypatch.setattr("surface_code.simulation.cli.run_aer", missing_aer)
    assert main(["run"]) == 1
    assert ".[sim]" in capsys.readouterr().err

    def unrelated_import(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'surprise'", name="surprise")

    monkeypatch.setattr("surface_code.simulation.cli.run_aer", unrelated_import)
    with pytest.raises(ModuleNotFoundError, match="surprise"):
        main(["run"])


def test_algebraic_commands_do_not_need_aer(monkeypatch, capsys):
    monkeypatch.setattr(
        "surface_code.simulation.cli.run_aer",
        lambda *args, **kwargs: pytest.fail("Aer should not run"),
    )
    assert main(["syndrome", "X4"]) == 0
    assert "Correction:" in capsys.readouterr().out
    assert main(["decode", "0000 0000", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["correction"] == "I"
    assert main(["info", "--json"]) == 0
    assert json.loads(capsys.readouterr().out)["parameters"] == {"distance": 3, "k": 1, "n": 9}


def test_sweep_reports_all_single_qubit_x_errors(capsys):
    assert main(["sweep", "--axes", "X"]) == 0
    output = capsys.readouterr().out
    assert "Cases:              9" in output
    assert "Decoded Z_L +1:     9" in output


def test_circuit_command_uses_requested_error(monkeypatch, capsys):
    class FakeCircuit:
        def draw(self, *, output):
            assert output == "text"
            return "fake circuit"

    seen = []

    def fake_to_qiskit(error):
        seen.append(error)
        return FakeCircuit()

    monkeypatch.setattr("surface_code.simulation.cli.to_qiskit", fake_to_qiskit)
    assert main(["circuit", "Z2"]) == 0
    assert seen == [Pauli.z_on((2,))]
    assert "fake circuit" in capsys.readouterr().out


def test_interactive_edits_do_not_measure_until_requested(monkeypatch, capsys):
    seen = []

    def fake_run(error, shots):
        seen.append((error, shots))
        return _fake_tallies(shots)

    monkeypatch.setattr("surface_code.simulation.cli.run_aer", fake_run)
    assert main(lines=["x 4", "measure", "quit"]) == 0
    assert seen == [(Pauli.x_on((4,)), 64)]
    assert "Decoded Z_L:  +1" in capsys.readouterr().out


def test_interactive_undo_and_duplicate_edits(monkeypatch):
    seen = []

    def fake_run(error, shots):
        seen.append(error)
        return _fake_tallies(shots)

    monkeypatch.setattr("surface_code.simulation.cli.run_aer", fake_run)
    assert main(lines=["z 3", "z 3", "undo", "measure", "q"]) == 0
    assert seen == [Pauli.z_on((3,))]


def test_interactive_reset_and_undo_restore_random_stream(monkeypatch, capsys):
    draws = []

    def fake_noise(p, qubits, rng):
        value = rng.random()
        draws.append(value)
        return Pauli.x_on((0,)) if value < 0.5 else Pauli.z_on((0,))

    monkeypatch.setattr("surface_code.simulation.cli.depolarizing_error", fake_noise)
    lines = ["draw", "reset", "draw", "undo", "draw", "q"]
    assert main(["interactive", "--seed", "7"], lines=lines) == 0
    assert draws == [draws[0]] * 3
    assert "draws=1" in capsys.readouterr().out


def test_interactive_invalid_input_recovers(capsys):
    assert main(lines=["x 99", "shots 0", "noise 2", "q"]) == 0
    captured = capsys.readouterr()
    assert captured.err.count("Error:") == 3


def test_interactive_prompt_defaults_accept_enter(capsys):
    assert main(lines=["shots", "", "rate", "", "q"]) == 0
    output = capsys.readouterr().out
    assert "Shots: 64" in output
    assert "Random-error probability: 0" in output


@pytest.mark.parametrize("lines", [["x"], ["shots"], ["rate"], ["add"]])
def test_interactive_eof_during_followup_prompt_is_clean(lines):
    assert main(lines=lines) == 0


def test_interactive_keyboard_interrupt_returns_130(monkeypatch):
    def interrupt(_prompt):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", interrupt)
    assert main(["interactive"]) == 130


def test_interactive_random_draw_is_explicit_and_measured_separately(monkeypatch):
    seen = []

    monkeypatch.setattr(
        "surface_code.simulation.cli.depolarizing_error",
        lambda p, qubits, rng=None: Pauli.z_on((3,)),
    )

    def fake_run(error, shots):
        seen.append(error)
        return _fake_tallies(shots)

    monkeypatch.setattr("surface_code.simulation.cli.run_aer", fake_run)
    assert main(lines=["rate 0.1", "draw", "measure", "q"]) == 0
    assert seen == [Pauli.z_on((3,))]


@pytest.mark.parametrize(
    "command",
    ["xl garbage", "draw garbage", "measure nonsense", "info garbage"],
)
def test_interactive_rejects_unexpected_arguments(command, capsys):
    assert main(lines=[command, "q"]) == 0
    assert "Error:" in capsys.readouterr().err


def test_interactive_circuit_accepts_an_error_override(monkeypatch):
    seen = []

    class FakeCircuit:
        def draw(self, *, output):
            return "circuit"

    def fake_to_qiskit(error):
        seen.append(error)
        return FakeCircuit()

    monkeypatch.setattr("surface_code.simulation.cli.to_qiskit", fake_to_qiskit)
    assert main(lines=["circuit X4", "q"]) == 0
    assert seen == [Pauli.x_on((4,))]


def test_missing_qiskit_does_not_close_interactive_session(monkeypatch, capsys):
    def missing(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'qiskit_aer'", name="qiskit_aer")

    monkeypatch.setattr("surface_code.simulation.cli.run_aer", missing)
    assert main(lines=["x 4", "measure", "state", "q"]) == 0
    captured = capsys.readouterr()
    assert "Aer needs Qiskit" in captured.err
    assert "Injected error: X4" in captured.out


def test_help_alias_and_command_typo(capsys):
    assert main(["help", "run"]) == 0
    assert "fixed injected error" in capsys.readouterr().out
    assert main(["syndrom", "X4"]) == 2
    assert "invalid choice" in capsys.readouterr().err


def test_malformed_pauli_shorthand_gets_a_pauli_error(capsys):
    assert main(["X9"]) == 2
    error = capsys.readouterr().err
    assert "data qubit must be" in error
    assert "invalid choice" not in error


def test_module_entry_point_shows_help():
    result = subprocess.run(
        [sys.executable, "-m", "surface_code", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "surface-code" in result.stdout
