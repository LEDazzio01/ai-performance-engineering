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

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)
from core.utils.compile_utils import configure_tf32, restore_tf32


class OptimizedGemmBenchmark(BaseBenchmark):
    """Single large matmul captured inside torch.compile."""

    def __init__(self):
        super().__init__()
        # Matrix dimensions (must match baseline for verification)
        self.m = 2048
        self.n = 2048
        self.k = 2048
        self.jitter_exemption_reason = "GEMM benchmark: fixed matrix dimensions"
        
        self.left: Optional[torch.Tensor] = None
        self.right: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.fn = None
        self._tf32_state = None
        
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )

    def setup(self) -> None:
        torch.manual_seed(1)
        self._tf32_state = configure_tf32(enable_matmul=True, enable_cudnn=True)
        
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

    def teardown(self) -> None:
        self.left = None
        self.right = None
        self.output = None
        self.fn = None
        if self._tf32_state is not None:
            restore_tf32(self._tf32_state)
            self._tf32_state = None
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

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is not None:
            return self.output.detach().clone()
        return torch.tensor([0.0], dtype=torch.float32, device=self.device)

    def get_output_tolerance(self) -> tuple:
        """Return custom tolerance for output comparison.
        
        The blocked baseline matmul accumulates results differently than this
        single matmul, leading to different floating-point rounding. With TF32
        enabled, this causes small numerical differences (~1e-3 absolute).
        
        Returns:
            Tuple of (rtol, atol)
        """
        return (1e-4, 1e-3)  # Single fused matmul has different FP accumulation order

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"m": self.m, "n": self.n, "k": self.k}


def get_benchmark() -> BaseBenchmark:
    return OptimizedGemmBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
