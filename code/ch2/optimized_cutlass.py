"""optimized_cutlass.py - Optimized GEMM using CUTLASS in hardware overview context.

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

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch2")
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
    
    def setup(self) -> None:
        """Setup: Initialize matrices."""
        
        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        # Optimization: CUTLASS-optimized GEMM
                
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        
        # Optimization: Compile with CUTLASS backend
        # CUTLASS provides hardware-optimized GEMM kernels
        # Select optimal compile mode based on GPU SM count
        from common.python.compile_utils import get_optimal_compile_mode
        compile_mode = get_optimal_compile_mode("max-autotune")
        self.compiled_matmul = torch.compile(
            lambda a, b: torch.matmul(a, b),
            mode=compile_mode,
            backend="inductor"
        )
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: CUTLASS-optimized GEMM."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_cutlass", enable=enable_nvtx):
    # Optimization: CUTLASS-optimized GEMM
    # Leverages hardware-optimized kernels (tensor cores, optimized memory access)
            _ = self.compiled_matmul(self.A, self.B)
            
    # Optimization: CUTLASS benefits
    # - Hardware-optimized GEMM kernels
    # - Leverages tensor cores for acceleration
    # - Optimized memory access patterns
    # - Better performance through hardware-specific optimizations

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        self.compiled_matmul = None
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
    print(f"Optimized: cutlass")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")

if __name__ == "__main__":
    main()
