"""Backwards-compatible shim that re-exports the shared tcgen05 loader."""

from common.tcgen05 import load_matmul_tcgen05_module

__all__ = ["load_matmul_tcgen05_module"]
