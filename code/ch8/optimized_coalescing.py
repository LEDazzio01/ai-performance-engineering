"""optimized_coalescing.py - Optimized coalesced memory access in occupancy/warp divergence context.

Demonstrates coalesced memory access patterns.
Coalescing: Uses coalesced memory access for efficient bandwidth utilization.
Threads access contiguous memory, maximizing bandwidth.
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
        raise RuntimeError("CUDA required for ch8")
    return torch.device("cuda")


class OptimizedCoalescingBenchmark(Benchmark):
    """Optimized: Coalesced memory access.
    
    Coalescing: Uses coalesced memory access for efficient bandwidth utilization.
    Threads access contiguous memory, maximizing bandwidth.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 1_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors with coalesced access pattern."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Coalesced memory access
        # Threads access contiguous memory for efficient coalescing
        # Coalescing: maximizes memory bandwidth utilization
        
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Coalesced memory access."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_coalescing", enable=enable_nvtx):
            # Optimization: Coalesced memory access
            # Contiguous access pattern enables coalescing
            # Coalescing: threads access contiguous memory for efficiency
            # All threads in warp access consecutive memory locations
            self.output = self.input * 2.0  # Contiguous access (coalesced)
            
            # Optimization: Coalescing benefits
            # - Contiguous memory access pattern
            # - Efficient memory bandwidth utilization
            # - Better performance through coalescing
            # - Maximized memory throughput

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input is None or self.output is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedCoalescingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedCoalescingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Coalescing")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

