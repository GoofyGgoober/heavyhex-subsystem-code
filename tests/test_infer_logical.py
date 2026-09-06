import pytest

from surface_code import Pauli
from surface_code.decoders.infer_logical import (
    logical_bit_from_data_readout,
    logical_readout_success,
    predict_logical_flip_from_history,
    syndrome_from_data_readout,
)
from surface_code.patches import PATCH
from surface_code.simulation.aer import shot_z_success


def _x_readout(error: Pauli) -> tuple[int, ...]:
    return tuple(int(qubit in error.x) for qubit in PATCH.data_qubits)


def test_history_separates_two_errors_that_defeat_final_only_decode():
    first_error = Pauli.x_on((0,))
    final_error = first_error * Pauli.x_on((3,))
    syndromes = (
        PATCH.code.syndrome(first_error),
        PATCH.code.syndrome(final_error),
    )
    data_bits = _x_readout(final_error)

    assert shot_z_success(syndromes[-1], data_bits) == 0
    assert logical_readout_success(
        syndromes,
        data_bits,
        basis="Z",
        interval_bit_error=0.01,
        syndrome_bit_error=0.01,
        terminal_bit_error=0.0,
    ) == 1


def test_one_round_syndrome_glitch_is_not_decoded_as_data_error():
    glitch = (0, 0, 0, 0, 1, 0, 0, 0)
    prediction = predict_logical_flip_from_history(
        (glitch, (0,) * 8),
        (0,) * 9,
        basis="Z",
        interval_bit_error=0.001,
        syndrome_bit_error=0.05,
        terminal_bit_error=0.0,
    )
    assert prediction.logical_flip == 0


def test_decoder_does_not_use_the_logical_measurement_to_choose_correction():
    zero = (0,) * 9
    logical_x_readout = _x_readout(PATCH.logical_x)
    kwargs = {
        "basis": "Z",
        "interval_bit_error": 0.01,
        "syndrome_bit_error": 0.01,
        "terminal_bit_error": 0.01,
    }
    assert syndrome_from_data_readout(zero, "Z") == syndrome_from_data_readout(
        logical_x_readout, "Z"
    )
    assert predict_logical_flip_from_history((), zero, **kwargs) == (
        predict_logical_flip_from_history((), logical_x_readout, **kwargs)
    )
    assert logical_bit_from_data_readout(zero, "Z") == 0
    assert logical_bit_from_data_readout(logical_x_readout, "Z") == 1


def test_terminal_boundary_recovers_a_final_readout_error():
    data_bits = _x_readout(Pauli.x_on((0,)))
    assert logical_readout_success(
        (),
        data_bits,
        basis="Z",
        interval_bit_error=0.0,
        syndrome_bit_error=0.0,
        terminal_bit_error=0.05,
    ) == 1


@pytest.mark.parametrize("basis", ["", "Y", "both"])
def test_decoder_rejects_unknown_basis(basis):
    with pytest.raises(ValueError, match="basis"):
        predict_logical_flip_from_history(
            (),
            (0,) * 9,
            basis=basis,
            interval_bit_error=0.0,
            syndrome_bit_error=0.0,
            terminal_bit_error=0.0,
        )
