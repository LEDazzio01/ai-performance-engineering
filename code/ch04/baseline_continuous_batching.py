"""Baseline continuous batching shim for ch04 multi-GPU wrappers.

Reuses the chapter 15 baseline implementation so the ch04 harness can discover
the benchmark and gracefully skip on single-GPU hosts.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from ch15.baseline_continuous_batching import BaselineContinuousBatchingBenchmark as _BaselineContinuousBatchingBenchmark


class BaselineContinuousBatchingBenchmark(_BaselineContinuousBatchingBenchmark):
    """Direct alias to the chapter 15 baseline; skips when GPUs < 2 in the wrapper."""


def get_benchmark() -> BaselineContinuousBatchingBenchmark:
    # Single-GPU runs are handled by the *multigpu wrapper raising SKIPPED",
    # but keep a safety check here to avoid accidental execution.
    if torch.cuda.device_count() < 2:
        raise RuntimeError("SKIPPED: baseline_continuous_batching requires >=2 GPUs")
    return BaselineContinuousBatchingBenchmark()
