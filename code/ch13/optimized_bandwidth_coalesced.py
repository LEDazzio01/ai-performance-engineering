"""optimized_bandwidth_coalesced.py - Coalesced bandwidth optimization (optimized).

Optimized memory access patterns with coalesced access.
Efficient bandwidth utilization through contiguous memory operations.

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


class OptimizedBandwidthCoalescedBenchmark(Benchmark):
    """Optimized bandwidth usage - coalesced memory access."""
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.C = None
        self.size = 10_000_000  # Same size for fair comparison
    
    def setup(self) -> None:
        """Setup: Initialize large tensors."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        
        # Large tensors for bandwidth measurement
        self.A = torch.randn(self.size, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.size, device=self.device, dtype=torch.float32)
        self.C = torch.empty_like(self.A)
        
        # Warmup
        self.C = self.A + self.B
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - optimized bandwidth usage."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_bandwidth_coalesced", enable=enable_nvtx):
            # Optimized pattern: coalesced contiguous access
            # Single vectorized operation achieves much better bandwidth
            self.C = self.A + self.B  # Coalesced access, single kernel
            
            # In-place operation avoids unnecessary memory transfer
            self.C.mul_(0.5)  # In-place multiply

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
    return OptimizedBandwidthCoalescedBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Bandwidth Coalesced: {result.mean_ms:.3f} ms")

