"""Decode recorded measurements by evolving a 32-entry probability table.

Each tick: A --new data errors W--> B --ancilla readout--> next A.
A table entry represents (four check bits, logical flip). Entries 0..15 have
no logical flip; entries 16..31 have a logical flip. The model assumes
independent data faults and independent errors in reported ancilla bits.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, TypeAlias, cast

from .._validation import validate_binary_bits, validate_probability
from ..patches import PATCH, Check

LogicalBasis: TypeAlias = Literal["X", "Z"]
Syndrome: TypeAlias = tuple[int, ...]
DataBits: TypeAlias = tuple[int, ...]

_NUM_SYNDROME_BITS = 4  # Four X-checks or four Z-checks.
_NUM_SYNDROMES = 2**_NUM_SYNDROME_BITS
_NUM_ERROR_CLASSES = 2 * _NUM_SYNDROMES  # Each syndrome, with or without a logical flip.


@dataclass(frozen=True)
class LogicalFlipPrediction:
    """Decoder prediction for the logical component of the correction."""

    logical_flip: int
    confidence: float


def predict_logical_flip_from_history(
    syndromes: Sequence[Syndrome],
    data_bits: DataBits,
    *,
    basis: str,
    interval_bit_error: float,
    syndrome_bit_error: float,
    terminal_bit_error: float,
) -> LogicalFlipPrediction:
    """Evolve A through the ancilla rounds, then predict the final logical flip.

    ``interval_bit_error``: relevant data error per qubit per tick.
    ``syndrome_bit_error``: wrong ancilla result per check.
    ``terminal_bit_error``: data/readout error after the last tick.
    Final data measurements supply check parities, not the logical answer.
    """
    measurement_basis = normalize_measurement_basis(basis)
    validate_probability("interval_bit_error", interval_bit_error)
    validate_probability("syndrome_bit_error", syndrome_bit_error)
    validate_probability("terminal_bit_error", terminal_bit_error)

    W = _build_data_error_table(interval_bit_error, measurement_basis)
    A = [0.0] * _NUM_ERROR_CLASSES
    A[0] = 1.0  # Initially: syndrome 0000, no logical flip.

    for ancilla_readout in syndromes:
        observed_syndrome = _syndrome_to_index(
            select_check_results(ancilla_readout, check_type=measurement_basis)
        )
        B = _predict_after_data_errors(A, W)
        A = _update_from_ancilla_readout(B, observed_syndrome, syndrome_bit_error)

    # Include errors after the last ancilla round, including final data readout.
    final_errors = _build_data_error_table(terminal_bit_error, measurement_basis)
    B = _predict_after_data_errors(A, final_errors)
    final_syndrome = _syndrome_to_index(syndrome_from_data_readout(data_bits, measurement_basis))
    # At the final syndrome, compare the no-flip and flip alternatives.
    logical_probabilities = (
        B[final_syndrome],
        B[final_syndrome + _NUM_SYNDROMES],
    )
    total = sum(logical_probabilities)
    if not total:
        raise ValueError("observations have zero probability under the decoder model")
    logical_flip = int(logical_probabilities[1] > logical_probabilities[0])
    return LogicalFlipPrediction(
        logical_flip=logical_flip,
        confidence=logical_probabilities[logical_flip] / total,
    )


def logical_readout_success(
    syndromes: Sequence[Syndrome],
    data_bits: DataBits,
    *,
    basis: str,
    interval_bit_error: float,
    syndrome_bit_error: float,
    terminal_bit_error: float,
) -> int:
    """Return 1 if history-based correction recovers the prepared +1 state, else 0."""
    prediction = predict_logical_flip_from_history(
        syndromes,
        data_bits,
        basis=basis,
        interval_bit_error=interval_bit_error,
        syndrome_bit_error=syndrome_bit_error,
        terminal_bit_error=terminal_bit_error,
    )
    return int(logical_bit_from_data_readout(data_bits, basis) == prediction.logical_flip)


def normalize_measurement_basis(basis: str) -> LogicalBasis:
    """Return uppercase X or Z; reject any other measurement basis."""
    normalized = basis.upper()
    if normalized not in {"X", "Z"}:
        raise ValueError(f"basis must be 'X' or 'Z', got {basis!r}")
    return cast(LogicalBasis, normalized)


def _checks_of_type(check_type: LogicalBasis) -> tuple[Check, ...]:
    return tuple(check for check in PATCH.checks if check.basis == check_type)


def _check_indices_of_type(check_type: str) -> tuple[int, ...]:
    return tuple(index for index, check in enumerate(PATCH.checks) if check.basis == check_type)


def _syndrome_to_index(syndrome: Sequence[int]) -> int:
    """Encode syndrome bits as an index, with the first bit least significant."""
    return sum(bit << index for index, bit in enumerate(syndrome))


def select_check_results(syndrome: Syndrome, *, check_type: str) -> tuple[int, ...]:
    """Four X- or Z-check bits from the eight-bit syndrome, in patch order."""
    normalized_check_type = check_type.upper()
    if normalized_check_type not in {"X", "Z"}:
        raise ValueError(f"check_type must be 'X' or 'Z', got {check_type!r}")
    validate_binary_bits("syndrome", syndrome, len(PATCH.checks))
    return tuple(syndrome[index] for index in _check_indices_of_type(normalized_check_type))


def syndrome_from_data_readout(data_bits: DataBits, basis: str) -> tuple[int, ...]:
    """Compute four X- or Z-check parities from final data-qubit measurements."""
    checked_basis = normalize_measurement_basis(basis)
    validate_binary_bits("data readout", data_bits, len(PATCH.data_qubits))
    return tuple(
        sum(data_bits[qubit] for qubit in check.support) % 2
        for check in _checks_of_type(checked_basis)
    )


def logical_bit_from_data_readout(data_bits: DataBits, basis: str) -> int:
    """Return the uncorrected logical measurement bit."""
    checked_basis = normalize_measurement_basis(basis)
    validate_binary_bits("data readout", data_bits, len(PATCH.data_qubits))
    logical = PATCH.logical_x if checked_basis == "X" else PATCH.logical_z
    support = logical.x if checked_basis == "X" else logical.z
    return sum(data_bits[qubit] for qubit in support) % 2


@lru_cache(maxsize=256)
def _build_data_error_table(bit_error: float, basis: LogicalBasis) -> tuple[float, ...]:
    """Build W: probabilities of new data errors, grouped into 32 error classes."""
    validate_probability("bit_error", bit_error)
    checks = _checks_of_type(basis)
    logical = PATCH.logical_x if basis == "X" else PATCH.logical_z
    logical_support = logical.x if basis == "X" else logical.z
    W = [0.0] * _NUM_ERROR_CLASSES
    num_data_qubits = len(PATCH.data_qubits)

    for error_mask in range(2**num_data_qubits):
        num_errors = error_mask.bit_count()
        pattern_probability = bit_error**num_errors * (1 - bit_error) ** (
            num_data_qubits - num_errors
        )
        syndrome_index = 0
        for index, check in enumerate(checks):
            parity = sum((error_mask >> qubit) & 1 for qubit in check.support) % 2
            syndrome_index |= parity << index
        logical_flip = sum((error_mask >> qubit) & 1 for qubit in logical_support) % 2
        error_class = syndrome_index + _NUM_SYNDROMES * logical_flip
        W[error_class] += pattern_probability
    return tuple(W)


def _predict_after_data_errors(A: Sequence[float], W: Sequence[float]) -> list[float]:
    """A -> B: combine accumulated errors with new data errors drawn from W."""
    B = [0.0] * _NUM_ERROR_CLASSES
    for previous_class, previous_probability in enumerate(A):
        if not previous_probability:
            continue
        for new_error_class, new_error_probability in enumerate(W):
            if new_error_probability:
                combined_class = previous_class ^ new_error_class
                B[combined_class] += previous_probability * new_error_probability
    return B


def _update_from_ancilla_readout(
    B: Sequence[float], observed_syndrome: int, readout_error: float
) -> list[float]:
    """B -> next A: multiply by readout likelihood L, then normalize."""
    A_next = [0.0] * _NUM_ERROR_CLASSES
    for error_class, probability in enumerate(B):
        true_syndrome = error_class % _NUM_SYNDROMES
        mismatched_bits = (true_syndrome ^ observed_syndrome).bit_count()
        L = readout_error**mismatched_bits * (1 - readout_error) ** (
            _NUM_SYNDROME_BITS - mismatched_bits
        )
        A_next[error_class] = probability * L
    normalization = sum(A_next)
    if not normalization:
        raise ValueError("observations have zero probability under the decoder model")
    for error_class in range(_NUM_ERROR_CLASSES):
        A_next[error_class] /= normalization
    return A_next
