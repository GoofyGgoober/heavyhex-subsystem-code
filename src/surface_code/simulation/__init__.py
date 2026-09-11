"""Simulators for the syndrome circuit. Aer is the first backend.

Run: python -m surface_code
"""

from .aer import run_aer, shot_z_success, to_qiskit
from .noise import depolarizing_error

__all__ = [
    "depolarizing_error",
    "run_aer",
    "shot_z_success",
    "to_qiskit",
]
