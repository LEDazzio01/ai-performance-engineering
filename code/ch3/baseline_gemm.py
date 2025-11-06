"""baseline_gemm.py - Baseline GEMM without optimization in infrastructure/OS tuning context.

Demonstrates standard matrix multiplication without optimization.
GEMM: General Matrix Multiply operation without optimization.
Uses standard PyTorch matmul without hardware-specific optimizations.
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


class BaselineGemmBenchmark(Benchmark):
    """Baseline: Standard GEMM without optimization.
    
    GEMM: General Matrix Multiply operation without optimization.
    Uses standard PyTorch matmul without hardware-specific optimizations.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        # Match optimized version size for fair comparison
        self.m = 4096
        self.n = 4096
        self.k = 4096
    
    def setup(self) -> None:
        """Setup: Initialize matrices."""
        torch.manual_seed(42)
        # Baseline: Standard GEMM (General Matrix Multiply)
        # GEMM is a fundamental operation for neural networks
        # This baseline uses standard PyTorch matmul without optimization
        
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard GEMM."""
        # Use conditional NVTX ranges - only enabled when profiling
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        with nvtx_range("baseline_gemm", enable=enable_nvtx):
            # Baseline: Standard GEMM (General Matrix Multiply)
            # GEMM: C = A @ B
            # Uses standard PyTorch matmul without optimization
            _ = torch.matmul(self.A, self.B)
            
            # Baseline: No GEMM optimization
            # Standard matrix multiplication without hardware-specific optimizations
    
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
    return BaselineGemmBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineGemmBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: GEMM")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

