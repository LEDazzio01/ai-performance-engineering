"""Helpers for gating Blackwell-only benchmarks."""

from __future__ import annotations

from common.python.cuda_capabilities import blackwell_tma_support_status


def ensure_blackwell_tma_supported(example_name: str) -> None:
    """Raise a SKIPPED error if Blackwell/GB-series GPUs are unavailable."""
    supported, reason = blackwell_tma_support_status()
    if not supported:
        raise RuntimeError(
            f"SKIPPED: {example_name} requires Blackwell/GB-series GPUs ({reason})"
        )
