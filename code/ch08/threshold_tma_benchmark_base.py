"""Shared utilities for TMA-gated threshold benchmarks."""

from __future__ import annotations

import torch

from core.benchmark.blackwell_requirements import ensure_blackwell_tma_supported

from ch08.threshold_benchmark_base import ThresholdBenchmarkBase


class ThresholdBenchmarkBaseTMA(ThresholdBenchmarkBase):
    """Threshold benchmark that only runs on Blackwell/GB-series GPUs."""

    requirement_label = "threshold_tma"
    rows: int = 1 << 25  # Larger working set for TMA microbenchmark
    threshold: float = 0.05
    inner_iterations: int = 16

    def setup(self) -> None:
        ensure_blackwell_tma_supported("Chapter 8 threshold TMA pipeline")
        super().setup()

    def _generate_inputs(self) -> torch.Tensor:  # type: ignore[override]
        return torch.empty(self.rows, device=self.device, dtype=torch.float32).uniform_(-1.0, 1.0)
