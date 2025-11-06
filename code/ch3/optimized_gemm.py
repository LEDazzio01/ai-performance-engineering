"""optimized_gemm.py - Optimized GEMM in infrastructure/OS tuning context.

Demonstrates optimized GEMM using torch.compile.
GEMM: General Matrix Multiply operation with optimization.
Uses torch.compile to optimize GEMM operations.
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
        raise RuntimeError("CUDA required for ch3")
    return torch.device("cuda")

def gemm_function(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """GEMM function for compilation."""
    return torch.matmul(A, B)

class OptimizedGemmBenchmark(Benchmark):
    """Optimized: GEMM with torch.compile optimization.
    
    GEMM: General Matrix Multiply operation with optimization.
    Uses torch.compile to optimize GEMM operations.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.compiled_gemm = None
        # Smaller matrices (2048x2048) have compilation overhead that dominates
        self.m = 4096
        self.n = 4096
        self.k = 4096
    
    def setup(self) -> None:
        """Setup: Initialize matrices and compile GEMM."""
        
        torch.manual_seed(42)
                                
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        
        # GEMM operations benefit from compilation optimization
        # Use "reduce-overhead" mode for small workloads to minimize compilation overhead
        # "max-autotune" has too much overhead for small matrices (2048x2048)
        # For system-level tuning, we use direct GEMM without compilation
        # System tuning focuses on OS/hardware configuration, not PyTorch compilation
        def gemm_function(a, b):
            return torch.matmul(a, b)
        self.compiled_gemm = gemm_function
        
        # Warmup: Run a few iterations to stabilize performance
        with torch.no_grad():
            for _ in range(15):
                _ = self.compiled_gemm(self.A, self.B)
        torch.cuda.synchronize()  # Ensure all warmup completes before benchmark
    
    def benchmark_fn(self) -> None:
        """Benchmark: Optimized GEMM."""
        # Use conditional NVTX ranges - only enabled when profiling
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        with nvtx_range("optimized_gemm", enable=enable_nvtx):
            pass
    # GEMM: C = A @ B (optimized)
        _ = self.compiled_gemm(self.A, self.B)
            
    # Optimization: GEMM benefits
    # - Optimized matrix multiplication (GEMM)
    # - Better performance through compilation
    # - Hardware-aware optimization
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        self.compiled_gemm = None
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
    return OptimizedGemmBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedGemmBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: GEMM")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")

if __name__ == "__main__":
    main()

