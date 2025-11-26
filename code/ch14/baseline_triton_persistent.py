"""Baseline Triton GEMM - Multiple kernel launches for batched operations.

This baseline demonstrates the cost of multiple kernel launches for
batched matrix multiplications, which persistent kernels optimize away.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import triton
import triton.language as tl
from typing import Optional

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


@triton.jit
def gemm_standard_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Standard GEMM: one block per output tile."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        mask_a = (offs_m[:, None] < M) & (offs_k[None, :] + k < K)
        mask_b = (offs_k[:, None] + k < K) & (offs_n[None, :] < N)
        
        a = tl.load(a_ptrs, mask=mask_a, other=0.0)
        b = tl.load(b_ptrs, mask=mask_b, other=0.0)
        
        acc += tl.dot(a, b)
        
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask_c = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask_c)


def matmul_standard_batched(a_batch: torch.Tensor, b_batch: torch.Tensor) -> torch.Tensor:
    """Multiple individual kernel launches for batched GEMM.
    
    This demonstrates the overhead of launching many small kernels.
    """
    batch_size = a_batch.shape[0]
    M, K = a_batch.shape[1], a_batch.shape[2]
    K, N = b_batch.shape[1], b_batch.shape[2]
    
    c_batch = torch.empty((batch_size, M, N), device=a_batch.device, dtype=a_batch.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 32, 32, 32
    
    # Launch separate kernel for each batch element (inefficient)
    for i in range(batch_size):
        a = a_batch[i]
        b = b_batch[i]
        c = c_batch[i]
        
        grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
        
        gemm_standard_kernel[grid](
            a, b, c,
            M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1),
            BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        )
    
    return c_batch


class BaselineTritonPersistentBenchmark(BaseBenchmark):
    """Baseline: Multiple kernel launches for batched operations.
    
    Demonstrates the overhead of launching many small kernels,
    which persistent/batched kernels optimize away.
    """

    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.batch_size = 32  # Many small operations
        self.M = 256
        self.N = 256
        self.K = 256
        self.dtype = torch.float16
        self._last = 0.0
        
        flops = 2 * self.batch_size * self.M * self.N * self.K
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.M * self.N),
        )

    def setup(self) -> None:
        """Setup: Initialize batched matrices."""
        torch.manual_seed(42)
        
        self.a = torch.randn(self.batch_size, self.M, self.K, device=self.device, dtype=self.dtype)
        self.b = torch.randn(self.batch_size, self.K, self.N, device=self.device, dtype=self.dtype)
        
        # Warmup
        for _ in range(3):
            _ = matmul_standard_batched(self.a, self.b)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: Multiple kernel launches."""
        output = matmul_standard_batched(self.a, self.b)
        self._last = float(output.sum())
        self._synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.a = None
        self.b = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
        )
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_triton_metrics
        return compute_triton_metrics(
            num_elements=getattr(self, 'N', getattr(self, 'num_elements', 1024)),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            block_size=getattr(self, 'BLOCK_SIZE', 1024),
            num_warps=getattr(self, 'num_warps', 4),
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.a is None or self.b is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return BaselineTritonPersistentBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline (Multiple Launches): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"  Config: batch={benchmark.batch_size}, M={benchmark.M}, N={benchmark.N}, K={benchmark.K}")
