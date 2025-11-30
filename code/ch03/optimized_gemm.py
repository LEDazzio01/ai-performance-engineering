"""Optimized GEMM that runs a fused tensor-core matmul in one launch."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn.functional as F

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)
from core.utils.compile_utils import enable_tf32


class OptimizedGemmBenchmark(BaseBenchmark):
    """Single large matmul captured inside torch.compile."""

    def __init__(self):
        super().__init__()
        self.m = 2048
        self.n = 2048
        self.k = 2048
        self.left: Optional[torch.Tensor] = None
        self.right: Optional[torch.Tensor] = None
        self.epilogue: Optional[torch.Tensor] = None
        self.fn = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )

    def setup(self) -> None:
        torch.manual_seed(1)
        enable_tf32()
        self.left = torch.randn(self.m, self.k, device=self.device, dtype=torch.float16)
        self.right = torch.randn(self.k, self.n, device=self.device, dtype=torch.float16)
        self.epilogue = torch.randn(self.m, self.n, device=self.device, dtype=torch.float16)

        def fused(a: torch.Tensor, b: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
            prod = torch.matmul(a, b)
            return F.gelu(prod + residual)

        compile_fn = getattr(torch, "compile", None)
        if compile_fn is not None:
            self.fn = compile_fn(fused, mode="reduce-overhead")
        else:
            raise RuntimeError("torch.compile is required for this benchmark")

        with torch.autocast("cuda", dtype=torch.float16):
            for _ in range(3):
                _ = self.fn(self.left, self.right, self.epilogue)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )

    def benchmark_fn(self) -> None:
        assert self.left is not None and self.right is not None and self.epilogue is not None
        op = self.fn
        with self._nvtx_range("optimized_gemm"):
            with torch.autocast("cuda", dtype=torch.float16):
                _ = op(self.left, self.right, self.epilogue)
        self._synchronize()

    def teardown(self) -> None:
        self.left = None
        self.right = None
        self.epilogue = None
        self.fn = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=10)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 1),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if self.fn is None:
            return "Fused function not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedGemmBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=10),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized GEMM latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
