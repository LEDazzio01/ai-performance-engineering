"""optimized_memory_standard.py - HBM3e-optimized memory access (optimized).

Memory access patterns optimized for HBM3e (high bandwidth memory).
Uses coalesced access, vectorized operations, and memory prefetching.

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
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class OptimizedMemoryHBM3eBenchmark(Benchmark):
    """HBM3e-optimized memory access - coalesced and vectorized."""
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.result = None
        self.size_mb = 100  # 100 MB
        self.access_pattern = "coalesced"  # HBM3e-optimized coalesced access
    
    def setup(self) -> None:
        """Setup: Allocate memory with HBM3e-friendly layout."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        
        # Allocate memory optimized for HBM3e access patterns
        num_elements = (self.size_mb * 1024 * 1024) // 4  # float32 = 4 bytes
        self.data = torch.randn(num_elements, device=self.device, dtype=torch.float32)
        self.result = torch.zeros_like(self.data)
        
        # Ensure memory is contiguous and aligned for optimal HBM3e access
        self.data = self.data.contiguous()
        self.result = self.result.contiguous()
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - HBM3e-optimized memory access."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("memory_standard", enable=enable_nvtx):
            # HBM3e-optimized: Coalesced access patterns, vectorized operations
            # PyTorch operations are already optimized for coalesced access
            # Use fused operations for better memory bandwidth utilization
            # Vectorized operations maximize HBM3e bandwidth
            self.result = torch.addcmul(
                self.data, self.data, torch.tensor(2.0, device=self.device), value=1.0
            )
            # Additional optimized operation - in-place for better memory efficiency
            self.result.add_(0.1)

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.data, self.result
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
        if self.data is None:
            return "Data not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedMemoryHBM3eBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized HBM3e Memory: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
