"""GPU requirement helpers for Chapter 8 benchmarks."""

from __future__ import annotations

import torch


def skip_if_insufficient_gpus(min_gpus: int = 2) -> None:
    """Raise a standardized SKIPPED RuntimeError when not enough GPUs exist."""
    available = torch.cuda.device_count()
    if available < min_gpus:
        raise RuntimeError(
            f"SKIPPED: Distributed benchmark requires multiple GPUs (found {available} GPU)"
        )
