"""Baseline benchmark for Chapter 13 warp specialization training example.

Standard PyTorch ops without warp specialization for comparison against Triton kernels."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineWarpSpecializationTrainingBenchmark(BaseBenchmark):
    """Baseline training workload without warp specialization."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.input: Optional[torch.Tensor] = None
        self.weight: Optional[torch.Tensor] = None
        self.batch = 512
        self.width = 2048
        tokens = self.batch * self.width
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Allocate tensors and build a simple MLP to mimic training compute."""
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(self.width, 4096),
            nn.GELU(),
            nn.Linear(4096, self.width),
        ).to(self.device).train()

        self.input = torch.randn(self.batch, self.width, device=self.device)
        self.weight = torch.randn_like(self.input)
        self._synchronize()

    def benchmark_fn(self) -> None:
        """Run the baseline forward pass (no warp specialization)."""
        if self.input is None or self.weight is None or self.model is None:
            raise RuntimeError("Benchmark not configured")

        with self._nvtx_range("baseline_warp_specialization_training"):
            with torch.no_grad():
                fused = torch.relu(self.input * self.weight)
                output = self.model(fused)
                _ = output.sum()
        self._synchronize()

    def teardown(self) -> None:
        """Release GPU resources."""
        self.model = None
        self.input = None
        self.weight = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        """Return a standard benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
            use_subprocess=False,
        )

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Sanity-check that buffers were initialized."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None or self.weight is None:
            return "Input tensors not initialized"
        return None


def get_benchmark() -> BaselineWarpSpecializationTrainingBenchmark:
    """Factory for harness discovery."""
    return BaselineWarpSpecializationTrainingBenchmark()
