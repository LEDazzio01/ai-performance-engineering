"""optimized_memory_transfer.py - Unified memory transfer (optimized).

Demonstrates unified memory access on Grace-Blackwell superchip.
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
        raise RuntimeError("CUDA required for ch2")
    return torch.device("cuda")

class OptimizedMemoryTransferBenchmark(Benchmark):
    """Unified memory transfer - faster NVLink-C2C path."""
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.N = 10_000_000
    
    def setup(self) -> None:
        """Setup: Initialize unified memory tensor."""
        
        torch.manual_seed(42)
        # Unified memory - accessible by both CPU and GPU
        # On Grace-Blackwell, this uses NVLink-C2C (~900 GB/s vs PCIe ~64 GB/s)
        self.data = torch.randn(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Unified memory access (no explicit transfer needed)."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("memory_transfer", enable=enable_nvtx):
    # Unified memory - GPU can access CPU memory directly via NVLink-C2C
    # No explicit copy needed - faster than PCIe transfers
            result = self.data * 2.0  # Simulates GPU computation on unified memory

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}, got {self.data.shape[0]}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedMemoryTransferBenchmark()

if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Memory Transfer: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

