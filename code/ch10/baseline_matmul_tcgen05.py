"""Baseline matmul benchmark: Custom tcgen05 kernel (educational).

CHAPTER 10 CONTEXT: This demonstrates a custom tcgen05 tensor core kernel.
The custom kernel is educational - it shows HOW tensor cores work and how
to write CUTLASS/CuTe kernels for Blackwell, but is NOT optimized for
maximum performance like cuBLAS.

Compare against: optimized_matmul_tcgen05.py (cuBLAS)
This pairing demonstrates that while custom kernels enable learning and
specialized optimizations, vendor libraries like cuBLAS are highly
optimized through years of engineering for general-purpose GEMM.
"""

from __future__ import annotations

from typing import Optional

import torch

from ch10.matmul_extension_tcgen05 import load_matmul_tcgen05_module
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.benchmark.tcgen05_requirements import check_tcgen05_support


class BaselineMatmulTCGen05Benchmark(BaseBenchmark):
    """Baseline: Custom tcgen05 CUDA kernel (educational implementation)."""

    def __init__(self) -> None:
        super().__init__()
        available, reason = check_tcgen05_support(
            loader=None,
            module_name="ch10 matmul tcgen05 kernels",
        )
        self._tcgen05_available = available
        self._skip_reason = reason or "SKIPPED: tcgen05 matmul unavailable"
        self.module = None
        self.device = torch.device("cuda")
        # Match other matmul benchmarks (baseline_matmul.py uses n=8192)
        self.n = 8192
        self.size = self.n
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(bytes_per_iteration=float(self.n * self.n * 2 * 3))

    def setup(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        if self.module is None:
            self.module = load_matmul_tcgen05_module()
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        dtype = torch.float16
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        assert self.A is not None and self.B is not None and self.module is not None
        with self._nvtx_range("baseline_matmul_tcgen05_custom"):
            with torch.no_grad():
                self.output = self.module.matmul_tcgen05(self.A, self.B)
        self._synchronize()

    def teardown(self) -> None:
        self.A = None
        self.B = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def validate_result(self) -> Optional[str]:
        if not self._tcgen05_available:
            return self._skip_reason
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification."""
        if self.output is None:
            raise RuntimeError("Output not available - run benchmark first")
        return self.output.detach().float()

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"size": self.size}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison - wider due to FP16."""
        return (0.5, 5.0)


def get_benchmark() -> BaselineMatmulTCGen05Benchmark:
    return BaselineMatmulTCGen05Benchmark()
