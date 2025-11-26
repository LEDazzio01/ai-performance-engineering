"""baseline_cublas.py - Naive FP32 matmul without TF32 tensor-core acceleration."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import configure_tf32, restore_tf32


class BaselineCublasBenchmark(BaseBenchmark):
    """
    Baseline: FP32 matmul with TF32 disabled.

    Demonstrates the cost of ignoring tensor-core friendly settings before we
    introduce a pure cuBLAS/TF32 path in the optimized example.
    """

    def __init__(self):
        super().__init__()
        self.m = 2048
        self.n = 2048
        self.k = 2048
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self._tf32_state: Optional[Tuple[Optional[str], Optional[str]]] = None
        tokens = self.m * self.n
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        """Allocate FP32 matrices and disable TF32 acceleration."""
        self._tf32_state = configure_tf32(matmul_precision="highest", cudnn_precision="highest")

        torch.manual_seed(42)
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        self._synchronize()

    def benchmark_fn(self) -> None:
        """Plain cuBLAS FP32 matmul."""
        assert self.A is not None and self.B is not None
        with self._nvtx_range("baseline_cublas_fp32"):
            _ = torch.matmul(self.A, self.B)
        self._synchronize()

    def teardown(self) -> None:
        """Restore TF32 settings and free tensors."""
        self.A = None
        self.B = None
        if self._tf32_state is not None:
            restore_tf32(self._tf32_state)
            self._tf32_state = None

        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)

    def get_workload_metadata(self):
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_memory_transfer_metrics
        return compute_memory_transfer_metrics(
            bytes_transferred=self._bytes_transferred if hasattr(self, '_bytes_transferred') else float(getattr(self, 'N', 1024) * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            transfer_type="hbm",
        )

    def validate_result(self) -> Optional[str]:
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineCublasBenchmark()
