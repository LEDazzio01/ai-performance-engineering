"""optimized_cutlass.py - Optimized GEMM using CUTLASS.

Demonstrates GEMM optimization using CUTLASS library for hardware-optimized kernels.
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

from common.python.cutlass_binding import cutlass_gemm_fp16


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch14")
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
        # Match baseline matrix size for fair comparison
        self.m = 4096
        self.n = 4096
        self.k = 4096
    
    def setup(self) -> None:
        """Setup: Initialize matrices with optimal configuration for CUTLASS."""
        torch.manual_seed(42)
        
        # Match baseline TF32 settings for fair comparison
        # Disable TF32 to isolate CUTLASS optimization effect (not TF32 vs non-TF32)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("high")
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Use float16 matrices for CUTLASS GEMM (matches baseline for fair comparison)
        # CUTLASS is optimized for FP16/Tensor Core acceleration
        # Same TF32 settings as baseline to isolate CUTLASS library effect
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        
        # Warmup the CUTLASS kernel to ensure kernels are cached before measurement
        _ = cutlass_gemm_fp16(self.A, self.B)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: CUTLASS-optimized GEMM."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_cutlass", enable=enable_nvtx):
            # Optimization: CUTLASS-optimized GEMM kernel
            # CUTLASS provides hardware-optimized kernels leveraging Tensor Cores
            # Same FP16 precision and TF32 settings as baseline for fair comparison
            # This isolates the CUTLASS library optimization effect
            if self.A is None or self.B is None:
                raise RuntimeError("Benchmark not initialized")
            _ = cutlass_gemm_fp16(self.A, self.B)
            torch.cuda.synchronize(self.device)
    
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
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedCutlassBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized CUTLASS: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
