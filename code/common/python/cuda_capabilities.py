"""Utilities for querying CUDA hardware feature support."""

from __future__ import annotations

from functools import lru_cache
from typing import Tuple
import os

try:
    import torch
except ImportError:  # pragma: no cover - torch always installed in benchmarks
    torch = None  # type: ignore[assignment]

# Global flag for forcing pipeline API (set via CLI flag, not env var)
_force_pipeline: bool = False


def set_force_pipeline(force: bool) -> None:
    """Set the force pipeline flag (replaces AIPERF_FORCE_PIPELINE env var).
    
    Args:
        force: If True, force enable pipeline kernels even on compute capability 12.0+.
    """
    global _force_pipeline
    _force_pipeline = force


@lru_cache(maxsize=1)
def _get_device_properties():
    """Return current CUDA device properties or None if unavailable."""
    if torch is None or not torch.cuda.is_available():
        return None
    device = torch.cuda.current_device()
    return torch.cuda.get_device_properties(device)


def _get_compute_capability() -> Tuple[int, int]:
    """Return (major, minor) compute capability, defaulting to (0, 0)."""
    props = _get_device_properties()
    if props is None:
        return (0, 0)
    return props.major, props.minor


def pipeline_support_status() -> Tuple[bool, str]:
    """Return (supported, reason) for CUDA Pipeline API availability."""
    major, minor = _get_compute_capability()
    if major == 0 and minor == 0:
        return False, "CUDA device not available"
    if major >= 8:
        return True, f"compute capability {major}.{minor} (>= 8.0)"
    return False, f"requires compute capability >= 8.0, found {major}.{minor}"


def supports_cuda_pipeline_api() -> bool:
    """Shortcut for pipeline API availability."""
    return pipeline_support_status()[0]


def tma_support_status() -> Tuple[bool, str]:
    """Return (supported, reason) for Tensor Memory Accelerator availability."""
    major, minor = _get_compute_capability()
    if major == 0 and minor == 0:
        return False, "CUDA device not available"
    if major >= 9:
        return True, f"compute capability {major}.{minor} (>= 9.0)"
    return False, f"requires compute capability >= 9.0 (Hopper+), found {major}.{minor}"


def supports_tensor_memory_accelerator() -> bool:
    """Shortcut for Tensor Memory Accelerator availability."""
    return tma_support_status()[0]


def blackwell_tma_support_status() -> Tuple[bool, str]:
    """Return (supported, reason) for Blackwell-ready TMA pipelines."""
    major, minor = _get_compute_capability()
    if major == 0 and minor == 0:
        return False, "CUDA device not available"
    if major >= 12:
        return True, f"compute capability {major}.{minor} (Blackwell/GB series)"
    return False, f"requires compute capability >= 12.0 (Blackwell), found {major}.{minor}"


def pipeline_runtime_allowed() -> Tuple[bool, str]:
    """Return (allowed, reason) for actually running pipeline kernels safely.
    
    This function checks if the CUDA Pipeline API can be safely used at runtime.
    On compute capability 12.0+ (Blackwell), pipeline kernels are disabled by
    default due to known instability issues. This can be overridden via the
    --force-pipeline CLI flag (or set_force_pipeline() function).
    
    Returns:
        Tuple of (allowed: bool, reason: str) indicating whether pipeline
        kernels can be safely run and the reason for the decision.
    """
    supported, reason = pipeline_support_status()
    if not supported:
        return False, reason
    # Check global flag (set via CLI flag)
    # Note: AIPERF_FORCE_PIPELINE env var is deprecated - use --force-pipeline CLI flag instead
    force = _force_pipeline
    major, minor = _get_compute_capability()
    if major >= 12 and not force:
        return True, (
            f"compute capability {major}.{minor} (automatic enablement on Blackwell/GB series)"
        )
    return True, f"compute capability {major}.{minor}"
