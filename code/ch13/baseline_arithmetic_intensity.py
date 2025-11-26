"""baseline_low_ai.py - Low arithmetic intensity baseline (baseline).

Memory-bound kernel with low arithmetic intensity.
Many memory operations relative to compute operations.

Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch


from typing import Optional

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class BaselineArithmeticIntensityBenchmark(BaseBenchmark):
    """Low arithmetic intensity baseline - memory-bound."""
    
    def __init__(self):
        super().__init__()
        self.A: torch.Tensor | None = None
        self.B: torch.Tensor | None = None
        self.C: torch.Tensor | None = None
        self.M = 2048
        self.K = 2048
        self.N = 2048
        self.block_k = 128  # Small tiles -> repeated memory traffic
        tokens = self.M * self.N
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
    
    def setup(self) -> None:
        """Setup: Initialize large tensors."""
        torch.manual_seed(42)
        
        # Allocate matrices for chunked matmul accumulation.
        self.A = torch.randn(self.M, self.K, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.K, self.N, device=self.device, dtype=torch.float32)
        self.C = torch.zeros(self.M, self.N, device=self.device, dtype=torch.float32)

        # Warm up chunked kernel launches.
        self._chunked_matmul()
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def _chunked_matmul(self) -> None:
        """Compute C = A @ B using small K-tiles."""
        assert self.A is not None and self.B is not None and self.C is not None
        self.C.zero_()
        for k in range(0, self.K, self.block_k):
            k_end = min(k + self.block_k, self.K)
            a_slice = self.A[:, k:k_end]
            b_slice = self.B[k:k_end, :]
            # Smaller matmuls yield low arithmetic intensity due to repeated reads/writes.
            self.C.addmm_(a_slice, b_slice, beta=1.0, alpha=1.0)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - low arithmetic intensity (memory-bound)."""
        with self._nvtx_range("baseline_arithmetic_intensity"):
            if self.A is None or self.B is None or self.C is None:
                raise RuntimeError("Benchmark not initialized")
            self._chunked_matmul()
            self._synchronize()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.C
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=25,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.C is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineArithmeticIntensityBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Arithmetic Intensity: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
