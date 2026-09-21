"""Aer-backed heavy-hex simulators."""

from .aer import (
    flagged_syndrome,
    run_flagged_circuit,
    run_memory,
    run_memory_flagged,
    syndrome_from_checks,
)
from .noise import depolarizing_error

__all__ = [
    "depolarizing_error",
    "flagged_syndrome",
    "run_flagged_circuit",
    "run_memory",
    "run_memory_flagged",
    "syndrome_from_checks",
]
