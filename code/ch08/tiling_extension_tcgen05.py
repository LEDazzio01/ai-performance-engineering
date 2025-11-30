"""Backwards-compatible shim that re-exports the shared tcgen05 loader."""

from core.common.tcgen05 import load_tiling_tcgen05_module

__all__ = ["load_tiling_tcgen05_module"]
