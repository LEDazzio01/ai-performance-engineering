"""optimized_stream_ordered.py - Stream-ordered memory allocation (optimized).

Demonstrates CUDA stream-ordered memory allocator (cudaMallocAsync) pattern.
Note: PyTorch uses its own allocator, but we demonstrate the stream-ordered pattern.
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
from common.python.stream_ordered import run_stream_ordered_allocator


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


class OptimizedStreamOrderedBenchmark(Benchmark):
    """Stream-ordered memory allocation - optimized pattern.
    
    Demonstrates stream-ordered allocation pattern where memory is allocated
    asynchronously on specific streams, enabling better overlap and reduced
    synchronization overhead.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.N = 5_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors and streams."""
        torch.manual_seed(42)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Stream-ordered memory allocation pattern."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("stream_ordered", enable=enable_nvtx):
            # REAL cudaMallocAsync workload with per-stream allocation
            run_stream_ordered_allocator(self.N, iterations=3)
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedStreamOrderedBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Stream Ordered (Async Allocation): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
