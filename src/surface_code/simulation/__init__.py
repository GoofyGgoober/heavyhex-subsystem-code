"""Simulators for the syndrome circuit. Aer is the first backend.

Run: python -m surface_code
"""

from .aer import run_aer, shot_z_success, to_qiskit

__all__ = [
    "run_aer",
    "shot_z_success",
    "to_qiskit",
]
