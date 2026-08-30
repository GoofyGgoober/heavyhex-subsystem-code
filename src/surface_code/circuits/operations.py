"""Minimal circuit operations used by syndrome extraction."""

from dataclasses import dataclass


@dataclass(frozen=True)
class H:
    qubit: int


@dataclass(frozen=True)
class CX:
    control: int
    target: int


@dataclass(frozen=True)
class MeasureZ:
    qubit: int
