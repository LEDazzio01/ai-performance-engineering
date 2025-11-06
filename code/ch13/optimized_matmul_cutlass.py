"""optimized_matmul_cutlass.py - CUTLASS GEMM optimization (optimized).

CUTLASS-optimized matrix multiplication using torch.compile for kernel fusion.
Leverages optimized GEMM kernels for better performance.

Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

# Import arch_config to apply Triton patch for sm_12x support
# The patch removes 'a' suffix from sm_121a -> sm_121 for ptxas compatibility
try:
    import arch_config  # noqa: F401
except ImportError:
    pass  # Continue if arch_config not available
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


def optimized_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """CUTLASS-optimized matrix multiplication via torch.compile."""
    # torch.compile automatically uses optimized kernels (including CUTLASS) when available
    return torch.matmul(A, B)


class OptimizedMatmulCUTLASSBenchmark(Benchmark):
    """CUTLASS matmul optimization - compiled GEMM."""
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.C = None
        self.compiled_matmul = None
        self.m = 2048
        self.n = 2048
        self.k = 2048
    
    def setup(self) -> None:
        """Setup: Initialize matrices and compile matmul."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        self.C = torch.empty(self.m, self.n, device=self.device, dtype=torch.float16)
        
        # Compile matmul function for CUTLASS optimization
        try:
            self.compiled_matmul = torch.compile(optimized_matmul, mode="reduce-overhead")
        except Exception:
            # Fallback to uncompiled if compilation fails
            self.compiled_matmul = optimized_matmul
        
        # Warmup (includes compilation if needed)
        for _ in range(10):
            _ = self.compiled_matmul(self.A, self.B)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - CUTLASS-optimized matmul."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_matmul_cutlass", enable=enable_nvtx):
            # CUTLASS-optimized matrix multiplication (via torch.compile)
            self.C = self.compiled_matmul(self.A, self.B)

    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.C, self.compiled_matmul
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
    return OptimizedMatmulCUTLASSBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized CUTLASS Matmul: {result.mean_ms:.3f} ms")

