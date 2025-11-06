"""baseline_cutlass.py - Baseline GEMM without CUTLASS in memory access/GEMM context.

Demonstrates standard matrix multiplication without CUTLASS optimization.
CUTLASS: This baseline does not use CUTLASS library for GEMM operations.
Uses standard PyTorch operations without hardware-specific optimizations.
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
        raise RuntimeError("CUDA required for ch7")
    return torch.device("cuda")


class BaselineCutlassBenchmark(Benchmark):
    """Baseline: Standard GEMM without CUTLASS.
    
    CUTLASS: This baseline does not use CUTLASS library for GEMM operations.
    Uses standard PyTorch matmul without hardware-specific optimizations.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.m = 2048
        self.n = 2048
        self.k = 2048
    
    def setup(self) -> None:
        """Setup: Initialize matrices."""
        torch.manual_seed(42)
        # Baseline: Standard PyTorch matmul - no CUTLASS optimization
        # CUTLASS provides hardware-optimized GEMM kernels
        # This baseline does not use CUTLASS
        
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard GEMM without CUTLASS."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_cutlass", enable=enable_nvtx):
            # Baseline: Standard PyTorch matmul (no CUTLASS)
            # Does not leverage CUTLASS hardware-optimized kernels
            # Standard computation path without hardware-specific optimizations
            _ = torch.matmul(self.A, self.B)
            
            # Baseline: No CUTLASS benefits
            # Standard GEMM without hardware-specific optimizations

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineCutlassBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineCutlassBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Cutlass")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
