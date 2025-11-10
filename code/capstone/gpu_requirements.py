"""Capability checks for capstone benchmarks."""

from __future__ import annotations

from common.python.tcgen05_requirements import ensure_tcgen05_supported as _ensure

try:
    from capstone.capstone_extension_tcgen05 import load_tcgen05_module
except Exception:  # pragma: no cover - optional dependency
    load_tcgen05_module = None


def ensure_tcgen05_supported() -> None:
    """Raise a SKIPPED error if tcgen05 kernels cannot run."""
    if load_tcgen05_module is None:
        raise RuntimeError(
            "SKIPPED: tcgen05 extension tooling is unavailable in this build."
        )
    _ensure(loader=load_tcgen05_module, module_name="capstone tcgen05 kernels")
