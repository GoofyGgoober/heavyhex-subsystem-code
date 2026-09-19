"""Heavy-hex simulators. Aer is the first backend.

Run: python -m surface_code
"""

from .heavyhex_aer import (
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
