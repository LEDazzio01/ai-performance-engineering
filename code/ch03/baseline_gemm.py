"""Baseline GEMM that serializes micro-batches with CPU synchronization.

This benchmark demonstrates inefficient kernel scheduling - splitting a large
matmul into many small operations with CPU synchronization between each.
The optimized version shows how a single fused operation is faster.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from core.utils.compile_utils import configure_tf32, restore_tf32


class BaselineGemmBenchmark(BaseBenchmark):
    """Splits a large GEMM into many small kernels with extra CPU sync."""

    def __init__(self):
        super().__init__()
        # Matrix dimensions (must match optimized for verification)
        self.m = 2048
        self.n = 2048
        self.k = 2048
        self.jitter_exemption_reason = "GEMM benchmark: fixed matrix dimensions"
        # Micro-batch size for blocked computation
        self.block_size = 256
        self.num_blocks = self.k // self.block_size
        
        self.left: Optional[torch.Tensor] = None
        self.right: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._tf32_state: Optional[Tuple[Optional[str], Optional[str]]] = None
        
        # Register workload metadata in __init__ for compliance checks
        self.register_workload_metadata(
            requests_per_iteration=float(self.num_blocks),
            tokens_per_iteration=float(self.m * self.n),
        )

    def setup(self) -> None:
        torch.manual_seed(1)
        self._tf32_state = configure_tf32(enable_matmul=True, enable_cudnn=True)
        
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

    def teardown(self) -> None:
        self.left = None
        self.right = None
        self.output = None
        if self._tf32_state is not None:
            restore_tf32(self._tf32_state)
            self._tf32_state = None
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

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is not None:
            return self.output.detach().clone()
        return torch.tensor([0.0], dtype=torch.float32, device=self.device)

    def get_output_tolerance(self) -> tuple:
        """Return custom tolerance for output comparison.
        
        The blocked matmul accumulates results differently than a single matmul,
        leading to different floating-point rounding. With TF32 enabled, this
        causes small numerical differences (~1e-3 absolute).
        
        Returns:
            Tuple of (rtol, atol)
        """
        return (1e-4, 1e-3)  # Blocked matmul has different FP accumulation order

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"m": self.m, "n": self.n, "k": self.k}


def get_benchmark() -> BaseBenchmark:
    return BaselineGemmBenchmark()
