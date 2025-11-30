"""optimized_matmul.py - Tensor Core optimized matrix multiplication.

Chapter 10 Optimization: This demonstrates how BF16/FP16 tensor core matmul
with cuBLAS is faster than tiled FP32 serial addmm operations. The baseline
deliberately uses inefficient tiled approach to show the benefit of:
1. Using reduced precision (BF16) for tensor core acceleration
2. Single fused matmul instead of serial tiled operations
3. TF32 enabled for additional speedup on Ampere+ GPUs
"""

from __future__ import annotations

from typing import Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.utils.compile_utils import enable_tf32


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


class OptimizedTensorCoreBenchmark(BaseBenchmark):
    """Optimized: Single BF16 matmul vs baseline's 64 tiled FP32 addmms.
    
    Demonstrates tensor core acceleration through:
    - BF16 precision for tensor core utilization
    - Single fused matmul operation instead of tiled approach
    - TF32 enabled for compute acceleration
    """
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.C = None
        self.n = 8192  # Match baseline workload signature
        self.tile_k = 128  # Match baseline for equivalent workload
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    
    def setup(self) -> None:
        """Setup: initialize matrices with same workload as baseline."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        
        torch.manual_seed(42)
        # Create FP32 matrices (same as baseline), then cast for optimized computation
        A_fp32 = torch.randn(self.n, self.n, device=self.device, dtype=torch.float32)
        B_fp32 = torch.randn(self.n, self.n, device=self.device, dtype=torch.float32)
        
        # Optimization: Cast to BF16 for tensor core acceleration
        self.A = A_fp32.to(self.dtype)
        self.B = B_fp32.to(self.dtype)
        self.C = torch.zeros(self.n, self.n, device=self.device, dtype=self.dtype)
        
        # Warmup to ensure cuBLAS kernels are loaded
        for _ in range(3):
            with torch.no_grad():
                _ = torch.matmul(self.A, self.B)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Optimized: Single fused BF16 matmul using tensor cores."""
        if self.A is None or self.B is None:
            raise RuntimeError("Matrices not initialized")
        
        with self._nvtx_range("matmul_tensor_core_optimized"):
            with torch.no_grad():
                # Single fused matmul - replaces 64 tiled addmm operations
                # BF16 enables tensor core acceleration on Ampere+ GPUs
                self.C = torch.matmul(self.A, self.B)
        self._synchronize()

    def teardown(self) -> None:
        """Cleanup."""
        self.A = None
        self.B = None
        self.C = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=10,
            warmup=5,  # Match baseline warmup
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None:
            return "Matrix A not initialized"
        if self.B is None:
            return "Matrix B not initialized"
        if self.A.shape != (self.n, self.n):
            return f"Matrix A shape mismatch: expected ({self.n}, {self.n}), got {self.A.shape}"
        if self.B.shape != (self.n, self.n):
            return f"Matrix B shape mismatch: expected ({self.n}, {self.n}), got {self.B.shape}"
        if not torch.isfinite(self.A).all():
            return "Matrix A contains non-finite values"
        if not torch.isfinite(self.B).all():
            return "Matrix B contains non-finite values"
        return None


def get_benchmark() -> OptimizedTensorCoreBenchmark:
    """Factory function for harness discovery."""
    return OptimizedTensorCoreBenchmark()
