"""optimized_high_ai.py - High arithmetic intensity optimization (optimized).

Compute-bound kernel with high arithmetic intensity.
Many compute operations relative to memory operations.
Optimized for maximum FLOPs per byte accessed.

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


class OptimizedArithmeticIntensityBenchmark(Benchmark):
    """High arithmetic intensity optimization - compute-bound."""
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.C = None
        self.size = 4096  # Smaller size but more compute per element
    
    def setup(self) -> None:
        """Setup: Initialize tensors for compute-bound operation."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        
        # Smaller tensors but with high compute intensity
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=torch.float32)
        self.C = torch.empty(self.size, self.size, device=self.device, dtype=torch.float32)
        
        # Warmup
        self.C = torch.matmul(self.A, self.B)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - high arithmetic intensity (compute-bound)."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_arithmetic_intensity", enable=enable_nvtx):
            # High arithmetic intensity: matrix multiplication
            # Few memory operations, many compute operations
            # MatMul: 2*N^3 FLOPs, 3*N^2 elements accessed
            # Arithmetic Intensity = 2*N^3 / (3*N^2*4 bytes) = 2*N / 12 = N/6
            # For N=4096: AI = 4096/6 ≈ 682 FLOPs/byte (compute-bound)
            self.C = torch.matmul(self.A, self.B)
            # Additional compute-heavy operations
            self.C = torch.matmul(self.C, self.A.T)
            self.C = torch.matmul(self.C, self.B.T)
            # High arithmetic intensity: many FLOPs per byte

    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.C
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
        if self.A is None:
            return "A not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedArithmeticIntensityBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Arithmetic Intensity: {result.mean_ms:.3f} ms")

