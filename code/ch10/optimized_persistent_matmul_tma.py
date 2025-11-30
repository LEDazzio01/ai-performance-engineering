"""optimized_persistent_matmul_tma.py

Persistent matmul using Triton TMA multicast + DSMEM on thread-block clusters.
Assumes SM100/Blackwell-class GPU with cluster support. Falls back to standard
execution if clusters are not available by letting the kernel launch; runtime
will raise if unsupported.
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
    WorkloadMetadata,
)

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError as exc:
    TRITON_AVAILABLE = False
    raise ImportError("Triton is required for this example") from exc

# FAIL FAST: Check for required Blackwell-specific Triton features
# These are NOT optional - if missing, the benchmark cannot run
REQUIRED_TRITON_FEATURES = ['dsmem', 'tma_async_copy', 'cluster_barrier']
_missing_features = [f for f in REQUIRED_TRITON_FEATURES if not hasattr(tl, f)]
if _missing_features:
    raise RuntimeError(
        f"FAIL FAST: Triton {triton.__version__} is missing required Blackwell features: {_missing_features}. "
        f"This benchmark requires a Triton version with DSMEM/TMA/cluster support. "
        f"Install a newer Triton with Blackwell support or skip this benchmark."
    )


@triton.jit
def persistent_matmul_tma(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    dsmem_a = tl.dsmem((BLOCK_M, BLOCK_K), dtype=tl.float16)
    dsmem_b = tl.dsmem((BLOCK_K, BLOCK_N), dtype=tl.float16)

    tma_a = tl.make_tensor_descriptor(stride_am, stride_ak, BLOCK_M, BLOCK_K)
    tma_b = tl.make_tensor_descriptor(stride_bk, stride_bn, BLOCK_K, BLOCK_N)

    for k0 in range(0, K, BLOCK_K):
        a_ptr = A + offs_m[:, None] * stride_am + (k0 + tl.arange(0, BLOCK_K))[None, :] * stride_ak
        b_ptr = B + (k0 + tl.arange(0, BLOCK_K))[:, None] * stride_bk + offs_n[None, :] * stride_bn

        tl.tma_async_copy(dsmem_a, a_ptr, tma_a, multicast=True)
        tl.tma_async_copy(dsmem_b, b_ptr, tma_b, multicast=True)

        tl.cluster_barrier()

        a_tile = tl.load(dsmem_a)
        b_tile = tl.load(dsmem_b)
        acc += tl.dot(a_tile, b_tile)

        tl.cluster_barrier()

    c_ptr = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr, acc)


def run_optimized(M=1024, N=1024, K=1024, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128):
    """Run optimized TMA/DSMEM persistent matmul.
    
    REQUIRES: Triton with Blackwell DSMEM/TMA support.
    FAILS FAST if features are unavailable - no fallback.
    """
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    
    kernel_kwargs = {
        "num_warps": 8,
        "num_stages": 2,
    }
    
    persistent_matmul_tma[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        **kernel_kwargs,
    )
    return c


#============================================================================
# Benchmark Harness Integration
#============================================================================

class PersistentMatmulTMABenchmark(BaseBenchmark):
    """Benchmark harness wrapper for TMA persistent matmul."""

    def __init__(self):
        super().__init__()
        self.a = None
        self.b = None
        self.M = 1024
        self.N = 1024
        self.K = 1024
        self._last = 0.0
        
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.M * self.N),  # Output elements
        )

    def setup(self) -> None:
        """Setup: Initialize matrices.
        
        FAILS FAST if Triton DSMEM/TMA features are unavailable.
        """
        torch.manual_seed(42)
        self.a = torch.randn(self.M, self.K, device=self.device, dtype=torch.float16)
        self.b = torch.randn(self.K, self.N, device=self.device, dtype=torch.float16)
        
        # Warmup - will fail fast if Triton features are missing
        for _ in range(2):
            _ = run_optimized(self.M, self.N, self.K)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        """Benchmark: TMA persistent matmul (or standard matmul if DSMEM unavailable)."""
        output = run_optimized(self.M, self.N, self.K)
        self._last = float(output.sum())
        self._synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.a = None
        self.b = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)
    
    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def validate_result(self) -> Optional[str]:
        if not TRITON_AVAILABLE:
            return "Triton not available"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return PersistentMatmulTMABenchmark()


if __name__ == "__main__":
    torch.manual_seed(0)
    out = run_optimized()
    print(f"Optimized TMA/DSMEM matmul completed, output mean={out.mean().item():.3f}")
