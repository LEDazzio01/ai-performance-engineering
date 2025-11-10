"""Shared helpers for thread block cluster benchmarks."""

from __future__ import annotations

CLUSTER_SKIP_HINTS = (
    "Thread block clusters unstable",
    "cluster target block not present",
    "cudaDevAttrClusterLaunch",
    "CUDA_EXCEPTION_17",
)


def should_skip_cluster_error(message: str) -> bool:
    """Return True if the runtime error indicates cluster hardware is unavailable."""
    msg_upper = message.upper()
    if "SKIPPED" in msg_upper:
        return True
    return any(hint in message for hint in CLUSTER_SKIP_HINTS)


def raise_cluster_skip(message: str) -> None:
    """Raise a standardized SKIPPED RuntimeError when cluster launch is unsupported."""
    if should_skip_cluster_error(message):
        raise RuntimeError(
            "SKIPPED: Thread block clusters unavailable on this driver/CUDA build "
            "(upgrade to CUDA 13.1+ or run under compute-sanitizer)."
        ) from None
