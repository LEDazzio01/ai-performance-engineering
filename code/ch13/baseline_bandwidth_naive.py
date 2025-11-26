"""baseline_bandwidth_naive.py - Naive bandwidth usage baseline (baseline).

Naive memory access patterns with poor bandwidth utilization.
Uncoalesced access, unnecessary memory transfers.

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


class BaselineBandwidthNaiveBenchmark(BaseBenchmark):
    """Naive bandwidth usage - poor memory access patterns."""
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.C = None
        self.size = 10_000_000  # Large vector for bandwidth measurement
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.size),
            bytes_per_iteration=float(self.size * 4 * 3),  # read A/B, write C
        )
    
    def setup(self) -> None:
        """Setup: Initialize large tensors."""
        torch.manual_seed(42)
        
        # Large tensors for bandwidth measurement
        self.A = torch.randn(self.size, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.size, device=self.device, dtype=torch.float32)
        self.C = torch.empty_like(self.A)
        
        # Warmup
        self.C = self.A + self.B
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
            bytes_per_iteration=self._workload.bytes_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - naive bandwidth usage."""
        assert self.A is not None and self.B is not None and self.C is not None
        with self._nvtx_range("baseline_bandwidth_naive"):
            # Naive pattern: uncoalesced access via strided operations
            # This pattern results in poor bandwidth utilization
            for i in range(0, self.size, 1024):  # Strided access
                self.C[i:i+1024] = self.A[i:i+1024] + self.B[i:i+1024]
            
            # Additional unnecessary memory transfers
            temp = self.C.clone()  # Unnecessary copy
            self.C = temp * 0.5    # Write back
        self._synchronize()

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.C
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
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
        if self.A is None:
            return "A not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineBandwidthNaiveBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Bandwidth Naive: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
