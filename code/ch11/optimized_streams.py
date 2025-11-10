"""optimized_streams.py - Concurrent kernel execution with streams (optimized).

Demonstrates overlapping kernel execution using CUDA streams.
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
        raise RuntimeError("CUDA required for ch11")
    return torch.device("cuda")


class OptimizedStreamsBenchmark(Benchmark):
    """Concurrent execution - kernels overlap.
    
    Note: For warp specialization examples, see optimized_streams_warp_specialized.cu
    which demonstrates warp specialization with __activemask for efficient stream-ordered execution.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.stream1 = None
        self.stream2 = None
        self.stream3 = None
        self.N = 5_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors and streams."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        self.data1 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.data2 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.data3 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        
        # Create separate streams for concurrent execution
        self.stream1 = torch.cuda.Stream()
        self.stream2 = torch.cuda.Stream()
        self.stream3 = torch.cuda.Stream()
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Concurrent kernel execution with streams."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("streams", enable=enable_nvtx):
            # Launch kernels on different streams - they can overlap
            with torch.cuda.stream(self.stream1):
                self.data1 = self.data1 * 2.0
            
            with torch.cuda.stream(self.stream2):
                self.data2 = self.data2 * 2.0
            
            with torch.cuda.stream(self.stream3):
                self.data3 = self.data3 * 2.0
            
            # Synchronize all streams
            self.stream1.synchronize()
            self.stream2.synchronize()
            self.stream3.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.stream1 = None
        self.stream2 = None
        self.stream3 = None
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
        if self.data1 is None:
            return "Data1 tensor not initialized"
        if self.data2 is None:
            return "Data2 tensor not initialized"
        if self.data3 is None:
            return "Data3 tensor not initialized"
        if not torch.isfinite(self.data1).all():
            return "Data1 contains non-finite values"
        if not torch.isfinite(self.data2).all():
            return "Data2 contains non-finite values"
        if not torch.isfinite(self.data3).all():
            return "Data3 contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedStreamsBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Streams: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")


