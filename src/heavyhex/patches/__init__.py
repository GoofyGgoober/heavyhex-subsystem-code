"""Heavy-hex patch definitions: operators and device embeddings."""

from .layout import HeavyHexLayout, build_layout
from .operators import D3, D5, HeavyHexOperators, build_operators

__all__ = [
    "D3",
    "D5",
    "HeavyHexLayout",
    "HeavyHexOperators",
    "build_layout",
    "build_operators",
]
