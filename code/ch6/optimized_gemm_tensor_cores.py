"""optimized_gemm_tensor_cores.py - Optimized GEMM with tensor cores and ILP.

Demonstrates matrix multiplication using tensor cores (WMMA/MMA) with ILP optimization.
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
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")

class OptimizedGEMMTensorCoresBenchmark(Benchmark):
    """Optimized GEMM - high ILP, uses tensor cores."""
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.C = None
        self.M = 512  # Matrix size
        self._roofline_metrics = None
    
    def setup(self) -> None:
        """Setup: Initialize matrices with FP16 for tensor cores."""
        
        # Enable TF32 for faster matmul on Ampere+ GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        # Use FP16/BF16 to enable tensor cores
        # Tensor cores are available on modern GPUs (V100+, A100, H100, B200)
        self.A = torch.randn(self.M, self.M, device=self.device, dtype=torch.float16)
        self.B = torch.randn(self.M, self.M, device=self.device, dtype=torch.float16)
        self.C = torch.empty(self.M, self.M, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()
        # Calculate roofline metrics
        self._calculate_roofline_metrics()
    
    def _calculate_roofline_metrics(self) -> None:
        """Calculate arithmetic intensity and roofline metrics."""
        # GEMM: C = A @ B
        # FLOPs: 2 * M^3 (M^2 multiplications + M^2 additions)
        flops = 2 * self.M ** 3
        # Bytes: Read A (M^2 * 2), Read B (M^2 * 2), Write C (M^2 * 2)
        # FP16 = 2 bytes
        bytes_accessed = 3 * self.M ** 2 * 2
        # Arithmetic intensity
        ai = flops / bytes_accessed
        self._roofline_metrics = {
            'flops': flops,
            'bytes': bytes_accessed,
            'ai': ai,
        }
    
    def benchmark_fn(self) -> None:
        """Benchmark: Optimized GEMM with tensor cores."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_gemm_tensor_cores", enable=enable_nvtx):
    # Optimization: Use FP16 matmul which uses tensor cores
    # PyTorch automatically uses tensor cores when available for FP16/BF16
    # This provides higher throughput and better ILP utilization
            self.C = torch.matmul(self.A, self.B)
    # Note: For custom CUDA kernels, we would use:
    # - wmma::load_matrix_sync for loading matrices
    # - wmma::mma_sync for matrix multiply-accumulate
    # - wmma::store_matrix_sync for storing results
    # This is handled automatically by PyTorch's optimized matmul

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.A = None
        self.B = None
        self.C = None
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
        if self.C is None:
            return "Output matrix C not initialized"
        if self.A is None or self.B is None:
            return "Input matrices not initialized"
        if self.C.shape != (self.M, self.M):
            return f"Output shape mismatch: expected ({self.M}, {self.M}), got {self.C.shape}"
        if not torch.isfinite(self.C).all():
            return "Output contains non-finite values"
        return None
    
    def get_roofline_metrics(self) -> dict:
        """Return roofline analysis metrics."""
        return self._roofline_metrics or {}

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedGEMMTensorCoresBenchmark()

if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    
    metrics = benchmark.get_roofline_metrics()
    print(f"\nOptimized GEMM Tensor Cores: {result.mean_ms:.3f} ms")
    print(f"Matrix size: {benchmark.M}×{benchmark.M}")
    print(f"Arithmetic Intensity: {metrics.get('ai', 0):.2f} FLOP/Byte")
    print(f"FLOPs: {metrics.get('flops', 0):.2e}")
    print(f"Bytes accessed: {metrics.get('bytes', 0):.2e}")
    print("\n  Tip: FP16 GEMM uses tensor cores (WMMA/MMA), providing 4-8x speedup over FP32")
    print("  Tensor cores provide higher ILP and better memory bandwidth utilization")
