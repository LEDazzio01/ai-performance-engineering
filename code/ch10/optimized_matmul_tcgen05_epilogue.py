"""Optimized TMEM epilogue (bias + SiLU) benchmark for tcgen05."""

from __future__ import annotations

from typing import Optional

import torch

from ch10.matmul_extension_tcgen05 import load_matmul_tcgen05_module
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from common.python.tcgen05_requirements import check_tcgen05_support


class OptimizedMatmulTCGen05EpilogueBenchmark(BaseBenchmark):
    """Runs the tcgen05 kernel with TMEM-resident bias + SiLU epilogue."""

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
        self.size = 4096
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        if self.module is None:
            self.module = load_matmul_tcgen05_module()
        torch.manual_seed(0)
        dtype = torch.float16
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        # Bias stored as float16 to match the GEMM inputs; epilogue promotes to float internally.
        self.bias = torch.randn(self.size, device=self.device, dtype=dtype)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        assert (
            self.A is not None
            and self.B is not None
            and self.bias is not None
            and self.module is not None
        )
        with self._nvtx_range("optimized_matmul_tcgen05_bias_silu"):
            with torch.no_grad():
                _ = self.module.matmul_tcgen05_bias_silu(self.A, self.B, self.bias)
        self._synchronize()

    def teardown(self) -> None:
        self.A = None
        self.B = None
        self.bias = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )
}

    def validate_result(self) -> Optional[str]:
        if not self._tcgen05_available:
            return self._skip_reason
        if self.A is None or self.B is None or self.bias is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> OptimizedMatmulTCGen05EpilogueBenchmark:
    return OptimizedMatmulTCGen05EpilogueBenchmark()
