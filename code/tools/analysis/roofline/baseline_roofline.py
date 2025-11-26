"""baseline_roofline.py - Baseline roofline without tensor cores."""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineRooflineBenchmark(BaseBenchmark):
    """Reads data with light compute to highlight bandwidth limits."""

    def __init__(self):
        super().__init__()
        self.model: Optional[nn.Module] = None
        self.data: Optional[torch.Tensor] = None
        self.batch_size = 32
        self.seq_len = 256
        self.hidden_dim = 256
        tokens = self.batch_size * self.seq_len * self.hidden_dim
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        ).to(self.device).eval()
        self.data = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim, device=self.device, dtype=torch.float32
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.model is not None and self.data is not None
        with self._nvtx_range("baseline_roofline"):
            with torch.no_grad():
                _ = self.model(self.data)
        self._synchronize()

    def teardown(self) -> None:
        self.model = None
        self.data = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=40, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics."""
        # Estimate problem size for roofline analysis
        n = getattr(self, 'N', 0) or getattr(self, 'hidden_dim', 0) or 4096
        batch = getattr(self, 'batch_size', 1) or getattr(self, 'batch', 1)
        # Simple FLOP estimate for linear layers
        flops = 2.0 * batch * n * n  # Rough estimate
        bytes_moved = batch * n * 4.0  # Input/output bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        return {
    "roofline.estimated_flops": flops,
    "roofline.estimated_bytes": bytes_moved,
    "roofline.arithmetic_intensity": arithmetic_intensity,
}

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.data is None:
            return "Model or data not initialized"
        return None


def get_benchmark() -> BaselineRooflineBenchmark:
    return BaselineRooflineBenchmark()
