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

from common.python.compile_utils import enable_tf32
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
        self.A: torch.Tensor | None = None
        self.B: torch.Tensor | None = None
        self.C: torch.Tensor | None = None
        self.M = 2048
        self.K = 2048
        self.N = 2048
    
    def setup(self) -> None:
        """Setup: Initialize tensors for compute-bound operation."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            enable_tf32()
        torch.manual_seed(42)
        
        # Smaller tensors but with high compute intensity
        self.A = torch.randn(self.M, self.K, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.K, self.N, device=self.device, dtype=torch.float32)
        self.C = torch.empty(self.M, self.N, device=self.device, dtype=torch.float32)

        # Warmup high-AI matmul so autotuning occurs before measurement.
        self._fast_matmul()
        torch.cuda.synchronize()

    def _fast_matmul(self) -> None:
        assert self.A is not None and self.B is not None and self.C is not None
        self.C = torch.matmul(self.A, self.B)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - high arithmetic intensity (compute-bound)."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_arithmetic_intensity", enable=enable_nvtx):
            if self.A is None or self.B is None or self.C is None:
                raise RuntimeError("Benchmark not initialized")
            # High arithmetic intensity: full fused matmul, single launch.
            self._fast_matmul()
            torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.C
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=25,
            warmup=5,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.C is None:
            return "Matrices not initialized"
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
    print(f"\nOptimized Arithmetic Intensity: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
