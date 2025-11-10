"""optimized_cutlass.py - Optimized GEMM using CUTLASS in memory access/GEMM context.

Demonstrates GEMM using CUTLASS library for hardware-optimized operations.
CUTLASS: Uses CUTLASS backend for hardware-optimized GEMM kernels.
Leverages tensor cores and optimized memory access patterns.
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
)
from common.python.compile_utils import enable_tf32
from common.python.cutlass_binding import cutlass_gemm_fp16


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch7")
    return torch.device("cuda")


class OptimizedCutlassBenchmark(Benchmark):
    """Optimized: GEMM using CUTLASS library.
    
    CUTLASS: Uses CUTLASS backend for hardware-optimized GEMM kernels.
    Leverages tensor cores and optimized memory access patterns for better performance.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.m = 2048
        self.n = 2048
        self.k = 2048
        self._cutlass_ready = False
    
    def setup(self) -> None:
        """Setup: Initialize matrices."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 using the new API to avoid legacy/new API mixing errors
            enable_tf32()
        torch.manual_seed(42)
        # Optimization: CUTLASS-optimized GEMM
        # Uses torch.compile with CUTLASS backend for hardware-optimized kernels
        
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        
        try:
            for _ in range(3):
                _ = cutlass_gemm_fp16(self.A, self.B)
            torch.cuda.synchronize()
            self._cutlass_ready = True
        except Exception as exc:
            raise RuntimeError(
                "CUTLASS GEMM extension unavailable; install nvidia-cutlass-dsl>=4.2."
            ) from exc
    
    def benchmark_fn(self) -> None:
        """Benchmark: CUTLASS-optimized GEMM."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_cutlass", enable=enable_nvtx):
            # Optimization: CUTLASS-optimized GEMM
            # Uses torch.compile with CUTLASS backend
            # Leverages hardware-optimized kernels (tensor cores, optimized memory access)
            if not self._cutlass_ready:
                raise RuntimeError("CUTLASS kernel not initialized")
            _ = cutlass_gemm_fp16(self.A, self.B)
            
            # Optimization: CUTLASS benefits
            # - Hardware-optimized GEMM kernels
            # - Leverages tensor cores for acceleration
            # - Optimized memory access patterns
            # - Better performance through hardware-specific optimizations

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        self._cutlass_ready = False
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
    return OptimizedCutlassBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedCutlassBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Cutlass")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
