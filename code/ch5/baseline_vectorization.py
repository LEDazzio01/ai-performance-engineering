"""baseline_vectorization.py - Baseline without vectorization in storage I/O context.

Demonstrates operations without vectorization optimization.
Vectorization: This baseline does not use vectorized operations.
Uses scalar operations which are inefficient.
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
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch5")
    return torch.device("cuda")


class BaselineVectorizationBenchmark(Benchmark):
    """Baseline: Scalar operations without vectorization.
    
    Vectorization: This baseline does not use vectorized operations.
    Uses scalar operations which are inefficient.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.N = 1_000_000
    
    def setup(self) -> None:
        """Setup: Initialize data."""
        torch.manual_seed(42)
        # Baseline: Scalar operations (no vectorization)
        # Vectorization processes multiple elements simultaneously
        # This baseline uses scalar operations
        
        self.data = torch.randn(self.N, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Scalar operations without vectorization."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_vectorization", enable=enable_nvtx):
            # Baseline: Scalar operations (no vectorization)
            # Processes elements one at a time (inefficient)
            # Vectorization would process multiple elements simultaneously
            result = torch.zeros(1, device=self.device)
            for i in range(min(1000, self.N)):  # Simulate scalar loop
                result += self.data[i]
            
            # Baseline: No vectorization benefits
            # Scalar operations (inefficient)
            _ = result

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineVectorizationBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineVectorizationBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Vectorization")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
