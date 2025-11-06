"""optimized_streams.py - Concurrent kernel execution with streams (optimized) in infrastructure/OS tuning context.

Demonstrates overlapping kernel execution using CUDA streams.
Streams: Uses CUDA streams for concurrent kernel execution.
Kernels overlap for better GPU utilization.
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
        raise RuntimeError("CUDA required for ch3")
    return torch.device("cuda")

class OptimizedStreamsBenchmark(Benchmark):
    """Concurrent execution - kernels overlap.
    
    Streams: Uses CUDA streams for concurrent kernel execution.
    Kernels overlap for better GPU utilization.
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
        
        torch.manual_seed(42)
        # Optimization: CUDA streams for concurrent execution
        # Streams allow kernels to execute concurrently
        # Better GPU utilization through overlapping execution
        
        self.data1 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.data2 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.data3 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        
        # Create separate streams for concurrent execution
        # Streams: enable overlapping kernel execution
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

        with nvtx_range("optimized_streams", enable=enable_nvtx):
    # Optimization: Concurrent execution with streams
    # Streams: launch kernels on different streams - they can overlap
    # Better GPU utilization through overlapping execution
            with torch.cuda.stream(self.stream1):
                self.data1 = self.data1 * 2.0
            
            with torch.cuda.stream(self.stream2):
                self.data2 = self.data2 * 2.0
            
            with torch.cuda.stream(self.stream3):
                self.data3 = self.data3 * 2.0
            
    # Synchronize all streams
            torch.cuda.synchronize()
            
    # Optimization: Streams benefits
    # - Concurrent kernel execution (streams)
    # - Better GPU utilization
    # - Overlapping execution for improved performance

    
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
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data1 is None:
            return "Data1 tensor not initialized"
        if self.data2 is None:
            return "Data2 tensor not initialized"
        if self.data3 is None:
            return "Data3 tensor not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedStreamsBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=30, warmup=5)
    )
    benchmark = OptimizedStreamsBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Streams")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")

if __name__ == "__main__":
    main()

