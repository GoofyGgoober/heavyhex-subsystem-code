"""Compat shim: the canonical builder lives in surface_code.patches.heavyhex_embed."""

from surface_code.patches.heavyhex_embed import (
    HeavyHexLayout,
    build_layout,
    operator_name,
)

__all__ = ["HeavyHexLayout", "build_layout", "operator_name"]
