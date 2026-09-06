"""Simulators for the syndrome circuit. Aer is the first backend.

Run: python -m surface_code
"""

from .aer import run_aer, shot_z_success, to_qiskit
from .sweep_n import (
    CadenceExperiment,
    CadencePoint,
    bare_qubit_failure_rate,
    run_cadence_experiment,
    run_full_cadence_experiment,
)
from .record_shots import (
    BASELINE_IDLE_NOISE,
    BASELINE_NOISE,
    PREPARATIONS,
    CircuitNoise,
    IdleNoise,
    MemoryShot,
    MemorySummary,
    idle_duration_per_round,
    run_memory,
    run_timed_memory,
    summarize_memory,
    to_memory_circuit,
    to_timed_memory_circuit,
)
from .noise import depolarizing_error
from .profiles import (
    BASELINE_PROFILE,
    GOOGLE_WILLOW_PROFILE,
    IBM_HERON_PROFILE,
    PROFILES,
    NoiseProfile,
    get_profile,
    idle_noise_from_t1_t2,
)

__all__ = [
    "BASELINE_IDLE_NOISE",
    "BASELINE_NOISE",
    "BASELINE_PROFILE",
    "GOOGLE_WILLOW_PROFILE",
    "IBM_HERON_PROFILE",
    "PREPARATIONS",
    "PROFILES",
    "CadenceExperiment",
    "CadencePoint",
    "CircuitNoise",
    "IdleNoise",
    "MemoryShot",
    "MemorySummary",
    "NoiseProfile",
    "bare_qubit_failure_rate",
    "depolarizing_error",
    "get_profile",
    "idle_duration_per_round",
    "idle_noise_from_t1_t2",
    "run_aer",
    "run_cadence_experiment",
    "run_full_cadence_experiment",
    "run_memory",
    "run_timed_memory",
    "shot_z_success",
    "summarize_memory",
    "to_memory_circuit",
    "to_qiskit",
    "to_timed_memory_circuit",
]
