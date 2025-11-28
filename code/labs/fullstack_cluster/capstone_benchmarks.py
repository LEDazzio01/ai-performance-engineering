from __future__ import annotations

import math
from typing import Callable, Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.fullstack_cluster import baseline_matmul


TensorFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class CapstoneMatmulBenchmark(BaseBenchmark):
    """Shared benchmark wrapper for the capstone GEMM kernels."""

    def __init__(
        self,
        runner: TensorFn,
        label: str,
        *,
        size: int = 2048,
        iterations: int = 3,
        warmup: int = 5,
        timeout_seconds: int = 300,
        validate_against_baseline: bool = True,
    ) -> None:
        super().__init__()
        self._runner = runner
        self._label = label
        self._size = size
        self._validate = validate_against_baseline
        self._last_output: Optional[torch.Tensor] = None
        self._reference: Optional[torch.Tensor] = None
        self._config = BenchmarkConfig(
            iterations=iterations,
            warmup=warmup,
            timeout_seconds=timeout_seconds,
            deterministic=False,
            enable_nvtx=False,
            enable_profiling=False,
        )
        tokens_per_iteration = float(size * size)
        flops_per_iteration = float(2 * (size ** 3))
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=tokens_per_iteration,
            samples_per_iteration=tokens_per_iteration,
            custom_units_per_iteration=flops_per_iteration,
            custom_unit_name="FLOPs",
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        self._lhs = torch.randn(
            self._size, self._size, device=self.device, dtype=torch.float16
        )
        self._rhs = torch.randn(
            self._size, self._size, device=self.device, dtype=torch.float16
        )
        torch.cuda.synchronize(self.device)

        if self._validate:
            self._reference = baseline_matmul(self._lhs, self._rhs).detach().clone()
            torch.cuda.synchronize(self.device)
        else:
            self._reference = None

    def benchmark_fn(self) -> None:
        with self._nvtx_range(self._label):
            self._last_output = self._runner(self._lhs, self._rhs)
        torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self._last_output = None
        super().teardown()

    def get_config(self) -> Optional[BenchmarkConfig]:
        return self._config

    def validate_result(self) -> Optional[str]:
        if not self._validate:
            return None
        if self._reference is None or self._last_output is None:
            return "Missing outputs for validation"
        diff = (self._last_output - self._reference).abs()
        max_diff = diff.max().item()
        if math.isnan(max_diff) or math.isinf(max_diff):
            return "Non-finite values detected in output tensor"
        # Empirically we see ~1e2 difference due to layout/pipeline differences.
        if max_diff > 200.0:
            return f"Max abs diff {max_diff:.2f} exceeds tolerance 200"
        return None

    def get_custom_metrics(self) -> Optional[dict]:
        """Return GEMM performance metrics for roofline analysis."""
        M = N = K = self._size
        flops = float(2 * M * N * K)  # MAD operations
        bytes_transferred = float((M * K + K * N + M * N) * 2)  # fp16
        return {
            f"{self._label}.size": float(self._size),
            f"{self._label}.flops": flops,
            f"{self._label}.bytes_transferred": bytes_transferred,
            f"{self._label}.arithmetic_intensity": flops / bytes_transferred if bytes_transferred > 0 else 0.0,
        }
