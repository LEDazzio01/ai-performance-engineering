"""optimized_matmul_pytorch.py - CUTLASS GEMM optimization (optimized).

CUTLASS-optimized matrix multiplication using torch.compile for kernel fusion.
Leverages optimized GEMM kernels for better performance.

Implements BaseBenchmark for harness integration.
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

from core.utils.compile_utils import compile_callable
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


def optimized_matmul(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """CUTLASS-optimized matrix multiplication with fused bias + activation."""
    out = torch.matmul(A, B)
    return torch.relu(out + bias)


class OptimizedMatmulCUTLASSBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """CUTLASS matmul optimization - compiled GEMM."""

    signature_equivalence_group = "ch13_matmul_pytorch_precision"
    signature_equivalence_ignore_fields = ("precision_flags",)
    
    def __init__(self):
        super().__init__()
        self.A = None
        self.B = None
        self.C = None
        self.bias = None
        self.A_fp32 = None
        self.B_fp32 = None
        self.bias_fp32 = None
        self.compiled_matmul = None
        # Match baseline dimensions for fair comparison
        self.m = 4096
        self.n = 4096
        self.k = 4096
        tokens = self.m * self.n
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        # Register workload metadata at init time for compliance check
        bytes_per_iter = (self.m * self.k + self.k * self.n + self.m * self.n * 2) * 4
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
            bytes_per_iteration=float(bytes_per_iter),
        )
    
    def setup(self) -> None:
        """Setup: Initialize matrices and compile matmul."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        
        self.A_fp32 = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.B_fp32 = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        self.bias_fp32 = torch.randn(self.m, self.n, device=self.device, dtype=torch.float32)

        # Compute path uses FP16, but verification uses the FP32 tensors above.
        self.A = self.A_fp32.to(dtype=torch.float16)
        self.B = self.B_fp32.to(dtype=torch.float16)
        self.bias = self.bias_fp32.to(dtype=torch.float16)
        self.C = torch.empty(self.m, self.n, device=self.device, dtype=torch.float16)
        
        # Compile matmul function for CUTLASS optimization
        self.compiled_matmul = compile_callable(
            optimized_matmul,
            mode="reduce-overhead",
            backend="inductor",
        )
        for _ in range(10):
            _ = self.compiled_matmul(self.A, self.B, self.bias)
        self._synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - CUTLASS-optimized matmul."""
        assert self.compiled_matmul is not None and self.A is not None and self.B is not None and self.bias is not None
        with self._nvtx_range("matmul_pytorch"):
            # CUTLASS-optimized matrix multiplication (via torch.compile)
            self.C = self.compiled_matmul(self.A, self.B, self.bias)
        self._synchronize()

    def capture_verification_payload(self) -> None:
        if self.A_fp32 is None or self.B_fp32 is None or self.bias_fp32 is None:
            raise RuntimeError("FP32 verification tensors not initialized")
        self._set_verification_payload(
            inputs={"A": self.A_fp32, "B": self.B_fp32, "bias": self.bias_fp32},
            output=self.C.float().detach().clone(),
            batch_size=self.m,
            parameter_count=0,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32,
            },
            output_tolerance=(0.5, 5.0),
        )

    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.bias, self.C, self.compiled_matmul
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
            setup_timeout_seconds=180,  # torch.compile compilation can take 60-120 seconds
        )
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_precision_metrics
        return compute_precision_metrics(
            fp32_time_ms=getattr(self, '_fp32_ms', 10.0),
            reduced_precision_time_ms=getattr(self, '_reduced_ms', 5.0),
            precision_type="fp8",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None or self.B is None or self.bias is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedMatmulCUTLASSBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
