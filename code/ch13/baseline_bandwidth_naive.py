"""baseline_bandwidth_naive.py - Naive bandwidth usage baseline (baseline).

Naive memory access patterns with poor bandwidth utilization.
Uncoalesced access, unnecessary memory transfers.

Implements Benchmark protocol for harness integration.
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
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class BaselineBandwidthNaiveBenchmark(Benchmark):
    """Naive bandwidth usage - poor memory access patterns."""
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.C = None
        self.size = 10_000_000  # Large vector for bandwidth measurement
    
    def setup(self) -> None:
        """Setup: Initialize large tensors."""
        torch.manual_seed(42)
        
        # Large tensors for bandwidth measurement
        self.A = torch.randn(self.size, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.size, device=self.device, dtype=torch.float32)
        self.C = torch.empty_like(self.A)
        
        # Warmup
        self.C = self.A + self.B
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - naive bandwidth usage."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_bandwidth_naive", enable=enable_nvtx):
            # Naive pattern: uncoalesced access via strided operations
            # This pattern results in poor bandwidth utilization
            for i in range(0, self.size, 1024):  # Strided access
                self.C[i:i+1024] = self.A[i:i+1024] + self.B[i:i+1024]
            
            # Additional unnecessary memory transfers
            temp = self.C.clone()  # Unnecessary copy
            self.C = temp * 0.5    # Write back

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.C
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None:
            return "A not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineBandwidthNaiveBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Bandwidth Naive: {result.mean_ms:.3f} ms")

