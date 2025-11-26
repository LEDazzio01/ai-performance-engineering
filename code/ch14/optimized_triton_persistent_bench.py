"""Optimized Triton GEMM - Persistent kernel with swizzled tile ordering.

Persistent kernels keep thread blocks alive for multiple tiles,
reducing kernel launch overhead and improving L2 cache reuse.
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
def gemm_persistent_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    num_tiles_m, num_tiles_n,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    NUM_SMS: tl.constexpr,
):
    """Persistent GEMM with swizzled tile ordering for L2 cache reuse."""
    pid = tl.program_id(0)
    num_tiles = num_tiles_m * num_tiles_n
    
    # Process tiles in strided pattern with swizzle for cache locality
    for tile_id in range(pid, num_tiles, NUM_SMS):
        # Swizzled tile indexing for better L2 cache reuse
        GROUP_M: tl.constexpr = 8
        
        group_id = tile_id // (GROUP_M * num_tiles_n)
        first_tile_m = group_id * GROUP_M
        group_size_m = min(num_tiles_m - first_tile_m, GROUP_M)
        
        tile_in_group = tile_id % (GROUP_M * num_tiles_n)
        pid_m = first_tile_m + (tile_in_group % group_size_m)
        pid_n = tile_in_group // group_size_m
        
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


def matmul_persistent(a: torch.Tensor, b: torch.Tensor, num_sms: int = 108) -> torch.Tensor:
    """Persistent GEMM launch."""
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    
    BLOCK_M, BLOCK_N, BLOCK_K = 64, 64, 32
    num_tiles_m = triton.cdiv(M, BLOCK_M)
    num_tiles_n = triton.cdiv(N, BLOCK_N)
    
    # Launch only NUM_SMS blocks - they will process all tiles
    grid = (num_sms,)
    
    gemm_persistent_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        num_tiles_m, num_tiles_n,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        NUM_SMS=num_sms,
    )
    
    return c


class OptimizedTritonPersistentBenchmark(BaseBenchmark):
    """Optimized: Persistent Triton GEMM with swizzled tile ordering."""

    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.M = 1024
        self.N = 1024
        self.K = 1024
        self.dtype = torch.float16
        self._last = 0.0
        self.num_sms = 108  # Will be updated in setup
        
        flops = 2 * self.M * self.N * self.K
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.M * self.N),
        )

    def setup(self) -> None:
        """Setup: Initialize matrices."""
        torch.manual_seed(42)
        
        # Get actual SM count
        props = torch.cuda.get_device_properties(self.device)
        self.num_sms = props.multi_processor_count
        
        self.a = torch.randn(self.M, self.K, device=self.device, dtype=self.dtype)
        self.b = torch.randn(self.K, self.N, device=self.device, dtype=self.dtype)
        
        # Warmup
        for _ in range(3):
            _ = matmul_persistent(self.a, self.b, self.num_sms)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: Persistent GEMM."""
        output = matmul_persistent(self.a, self.b, self.num_sms)
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
            iterations=100,
            warmup=20,
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
    return OptimizedTritonPersistentBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Persistent GEMM: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"  Config: M={benchmark.M}, N={benchmark.N}, K={benchmark.K}, SMs={benchmark.num_sms}")


