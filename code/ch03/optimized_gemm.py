"""Optimized GEMM that runs a fused tensor-core matmul in one launch.

This benchmark demonstrates efficient kernel scheduling - using torch.compile
to fuse operations and leverage tensor cores for a single large matmul.
The baseline version shows how splitting into micro-batches is slower.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


class OptimizedGemmBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Single large matmul captured inside torch.compile."""

    def __init__(self):
        super().__init__()
        # Matrix dimensions (must match baseline for verification)
        self.m = 2048
        self.n = 2048
        self.k = 2048
        
        self.left: Optional[torch.Tensor] = None
        self.right: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.fn = None
        
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Create input matrices - same as baseline version
        self.left = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.right = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        self.output = None

        def matmul_fn(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return torch.matmul(a, b)

        compile_fn = getattr(torch, "compile", None)
        if compile_fn is not None:
            self.fn = compile_fn(matmul_fn, mode="reduce-overhead")
        else:
            raise RuntimeError("torch.compile is required for this benchmark")

        # Warmup compiled function
        for _ in range(3):
            _ = self.fn(self.left, self.right)
        self._synchronize()

    def benchmark_fn(self) -> None:
        """Compute C = A @ B using a single fused matmul.
        
        This demonstrates efficient kernel scheduling where a single
        optimized operation is used instead of many small ones.
        """
        assert self.left is not None and self.right is not None
        op = self.fn
        
        with self._nvtx_range("optimized_gemm"):
            result = op(self.left, self.right)
        self._synchronize()
        
        self.output = result

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"left": self.left, "right": self.right},
            output=self.output.detach().clone(),
            batch_size=self.left.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": True,
            },
            output_tolerance=(1e-4, 1e-3),
        )

    def teardown(self) -> None:
        self.left = None
        self.right = None
        self.output = None
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
            return "Compiled function not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedGemmBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
