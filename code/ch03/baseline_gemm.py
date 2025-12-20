"""Baseline GEMM that serializes micro-batches with CPU synchronization.

This benchmark demonstrates inefficient kernel scheduling - splitting a large
matmul into many small operations with CPU synchronization between each.
The optimized version shows how a single fused operation is faster.
"""

from __future__ import annotations

from typing import List, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselineGemmBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Splits a large GEMM into many small kernels with extra CPU sync."""

    def __init__(self):
        super().__init__()
        # Matrix dimensions (must match optimized for verification)
        self.m = 2048
        self.n = 2048
        self.k = 2048
        # Micro-batch size for blocked computation
        self.block_size = 256
        self.num_blocks = self.k // self.block_size
        
        self.left: Optional[torch.Tensor] = None
        self.right: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.m * self.n),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        # Create input matrices - same as optimized version
        self.left = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.right = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        self.output = None
        self._synchronize()

    def benchmark_fn(self) -> None:
        """Compute C = A @ B using blocked micro-batches with CPU sync.
        
        This simulates poor kernel scheduling where we launch many small
        kernels with CPU synchronization between each, rather than a single
        large optimized operation.
        """
        from core.profiling.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        # Accumulate result from blocked matmul operations
        # C = A @ B = sum over blocks of (A[:, block_i] @ B[block_i, :])
        result = torch.zeros(self.m, self.n, device=self.device, dtype=torch.float32)
        
        with nvtx_range("baseline_gemm", enable=enable_nvtx):
            for i in range(self.num_blocks):
                start = i * self.block_size
                end = start + self.block_size
                # Extract block slices
                left_block = self.left[:, start:end]  # (m, block_size)
                right_block = self.right[start:end, :]  # (block_size, n)
                # Accumulate partial result
                result += torch.matmul(left_block, right_block)
                self._synchronize()  # CPU sync between blocks (inefficient!)
        
        self.output = result
        if self.left is None or self.right is None:
            raise RuntimeError("benchmark_fn() must be called after setup initializes inputs")

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
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 1),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if self.left is None or self.right is None:
            return "Input matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineGemmBenchmark()
