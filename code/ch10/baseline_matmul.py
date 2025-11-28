"""baseline_matmul.py - Baseline FP32 matmul with serial tiling."""

from __future__ import annotations

from typing import Optional

import torch

try:
    import ch10.arch_config  # noqa: F401
except ImportError:
    pass

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineMatmulBenchmark(BaseBenchmark):
    """Baseline: FP32 matmul with serialized tiling and no tensor cores."""

    def __init__(self):
        super().__init__()
        self.A: torch.Tensor | None = None
        self.B: torch.Tensor | None = None
        self.C: torch.Tensor | None = None
        self.n = 8192
        self.tile_k = 128

    def setup(self) -> None:
        """Setup: initialize FP32 matrices and scratch buffer."""
        torch.manual_seed(42)
        self.A = torch.randn(self.n, self.n, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.n, self.n, device=self.device, dtype=torch.float32)
        self.C = torch.zeros(self.n, self.n, device=self.device, dtype=torch.float32)
        self._chunked_matmul()
        self._synchronize()

    def _chunked_matmul(self) -> None:
        """Multiply using many FP32 tiles to emphasize poor reuse."""
        assert self.A is not None and self.B is not None and self.C is not None
        self.C.zero_()
        for k in range(0, self.n, self.tile_k):
            k_end = min(k + self.tile_k, self.n)
            a_tile = self.A[:, k:k_end]
            b_tile = self.B[k:k_end, :]
            # Repeated addmm launches mimic per-stage kernel launches.
            self.C.addmm_(a_tile, b_tile, beta=1.0, alpha=1.0)

    def benchmark_fn(self) -> None:
        """Benchmark: serialized FP32 matmul tiles."""
        if self.A is None or self.B is None or self.C is None:
            raise RuntimeError("Matrices not initialized")
        with self._nvtx_range("matmul_baseline_fp32"):
            self._chunked_matmul()
        self._synchronize()

    def teardown(self) -> None:
        """Teardown: clean up tensors."""
        self.A = None
        self.B = None
        self.C = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=5,  # Minimum warmup for accurate matmul timing
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.C is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaselineMatmulBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineMatmulBenchmark()
