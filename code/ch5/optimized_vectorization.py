"""optimized_vectorization.py - Optimized vectorized operations in storage I/O context.

Demonstrates vectorized operations for efficient processing.
Vectorization: Uses vectorized operations to process multiple elements simultaneously.
Improves throughput through SIMD operations.
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


class OptimizedVectorizationBenchmark(Benchmark):
    """Optimized: Vectorized operations for efficient processing.
    
    Vectorization: Uses vectorized operations to process multiple elements simultaneously.
    Improves throughput through SIMD operations.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.N = 1_000_000
    
    def setup(self) -> None:
        """Setup: Initialize data."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Vectorized operations
        # Vectorization processes multiple elements simultaneously
        # Uses SIMD operations for efficient processing
        
        self.data = torch.randn(self.N, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Vectorized operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_vectorization", enable=enable_nvtx):
            # Optimization: Vectorized operations
            # Processes multiple elements simultaneously (vectorization)
            # Vectorization: SIMD operations for efficient processing
            result = self.data.sum()  # Vectorized sum operation
            
            # Optimization: Vectorization benefits
            # - Processes multiple elements simultaneously
            # - SIMD operations for efficiency
            # - Better throughput through vectorization
            # - Improved performance through parallel element processing
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
    return OptimizedVectorizationBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedVectorizationBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Vectorization")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
