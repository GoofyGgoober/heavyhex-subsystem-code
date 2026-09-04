"""Simulators for the syndrome circuit. Aer is the first backend.

Run: python -m surface_code
"""

from .aer import run_aer, shot_z_success, to_qiskit
from .memory import (
    BASELINE_NOISE,
    CircuitNoise,
    MemoryShot,
    MemorySummary,
    run_memory,
    summarize_memory,
    to_memory_circuit,
)
from .noise import depolarizing_error

__all__ = [
    "BASELINE_NOISE",
    "CircuitNoise",
    "MemoryShot",
    "MemorySummary",
    "depolarizing_error",
    "run_aer",
    "run_memory",
    "shot_z_success",
    "summarize_memory",
    "to_memory_circuit",
    "to_qiskit",
]
