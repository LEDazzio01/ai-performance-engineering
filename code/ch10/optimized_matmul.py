"""optimized_matmul.py - Tensor Core optimized matrix multiplication."""

from __future__ import annotations

from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from common.python.compile_utils import enable_tf32


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")


def tensor_core_matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """Optimized matmul using tensor cores (FP16/BF16)."""
    if A.dtype != B.dtype:
        raise RuntimeError("Incompatible dtypes for tensor core GEMM")
    return torch.matmul(A, B)


class OptimizedTensorCoreBenchmark(BaseBenchmark):
    """Benchmark implementation using tensor cores."""
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.size = 8192
        self.dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        self.matmul_fn = tensor_core_matmul
    
    def setup(self) -> None:
        """Setup: initialize matrices."""
        
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
        compile_fn = getattr(torch, "compile", None)
        if callable(compile_fn):
            try:
                self.matmul_fn = compile_fn(tensor_core_matmul, mode="reduce-overhead")  # type: ignore[arg-type]
            except Exception:
                self.matmul_fn = tensor_core_matmul
        with torch.no_grad():
            _ = self.matmul_fn(self.A, self.B)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        if self.A is None or self.B is None:
            raise RuntimeError("Matrices not initialized")
        with self._nvtx_range("matmul_tensor_core"):
            with torch.no_grad():
                _ = self.matmul_fn(self.A, self.B)
        self._synchronize()

    def teardown(self) -> None:
        """Cleanup."""
        self.A = None
        self.B = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
            iterations=10,
            warmup=10,  # torch.compile requires extra warmup for JIT
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
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
        if self.A.shape != (self.size, self.size):
            return f"Matrix A shape mismatch: expected ({self.size}, {self.size}), got {self.A.shape}"
        if self.B.shape != (self.size, self.size):
            return f"Matrix B shape mismatch: expected ({self.size}, {self.size}), got {self.B.shape}"
        if not torch.isfinite(self.A).all():
            return "Matrix A contains non-finite values"
        if not torch.isfinite(self.B).all():
            return "Matrix B contains non-finite values"
        return None


def get_benchmark() -> OptimizedTensorCoreBenchmark:
    """Factory function for harness discovery."""
    return OptimizedTensorCoreBenchmark()
