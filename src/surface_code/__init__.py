"""Heavy-hex subsystem code on IBM hardware, built from scratch."""

from .circuits import memory_circuit, memory_circuit_flagged
from .core import Pauli, StabilizerCode, SubsystemCode
from .decoders import decode, residual_in_gauge_group
from .patches import D3, D5, HeavyHexOperators, build_operators
from .simulation import depolarizing_error, run_memory, run_memory_flagged

__all__ = [
    "D3",
    "D5",
    "HeavyHexOperators",
    "Pauli",
    "StabilizerCode",
    "SubsystemCode",
    "build_operators",
    "decode",
    "depolarizing_error",
    "memory_circuit",
    "memory_circuit_flagged",
    "residual_in_gauge_group",
    "run_memory",
    "run_memory_flagged",
]
