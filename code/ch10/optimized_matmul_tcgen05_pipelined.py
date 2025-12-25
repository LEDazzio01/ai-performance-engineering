"""Optimized matmul benchmark: Pipelined tcgen05 kernel variant.

CHAPTER 10 CONTEXT: Uses an advanced pipelined tcgen05 kernel from the
custom_vs_cublas lab to overlap compute and memory via double-buffering.

Key optimizations over basic tcgen05:
1. Double-buffered shared memory for async prefetch
2. Overlapped TMA loads with MMA compute
3. Reduced barrier waits (no-wait pipeline pattern)

Compare against:
- baseline_matmul_tcgen05.py (cuBLAS) - The gold standard
- optimized_matmul_tcgen05.py (basic tcgen05) - Single-stage version

EDUCATIONAL VALUE: Shows the progression from basic tensor core kernel
to a pipelined implementation with overlap.
"""

from __future__ import annotations

from typing import Optional

import torch

from ch10.matmul_extension_tcgen05 import load_matmul_tcgen05_module
from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.tcgen05_requirements import check_tcgen05_support
from labs.custom_vs_cublas.tcgen05_loader import matmul_tcgen05_no_wait


class OptimizedMatmulTCGen05PipelinedBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Pipelined tcgen05 kernel with double-buffering.
    """

    def __init__(self) -> None:
        super().__init__()
        available, reason = check_tcgen05_support(
            loader=None,
            module_name="ch10 matmul tcgen05 kernels",
        )
        self._tcgen05_available = available
        self._skip_reason = reason or "SKIPPED: tcgen05 matmul unavailable"
        self.device = torch.device("cuda")
        # Match baseline for fair comparison (larger size reduces CPU overhead noise).
        self.n = 12288
        self.size = self.n
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self._placeholder_kernel = False
        self._warned_placeholder = False
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(bytes_per_iteration=float(self.n * self.n * 2 * 3))

    def setup(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        dtype = torch.float16
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        assert self.A is not None and self.B is not None
        with self._nvtx_range("optimized_matmul_tcgen05_pipelined"):
            with torch.no_grad():
                self.output = matmul_tcgen05_no_wait(self.A, self.B)
        self._synchronize()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B},
            output=self.output.detach().float().clone(),
            batch_size=self.size,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(5e-2, 5e-2),
        )

    def teardown(self) -> None:
        self.A = None
        self.B = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics."""
        # Calculate theoretical TFLOPS
        flops = 2 * self.size ** 3  # GEMM is 2*M*N*K FLOPs
        return {
            "matrix_size": self.size,
            "theoretical_flops": flops,
            "optimization": "pipelined tcgen05 (no-wait async overlap)",
            "pipelined_kernel_available": True,
        }

    def validate_result(self) -> Optional[str]:
        if not self._tcgen05_available:
            return self._skip_reason
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> OptimizedMatmulTCGen05PipelinedBenchmark:
    return OptimizedMatmulTCGen05PipelinedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
