"""baseline_persistent_matmul_tma.py

Reference Triton matmul without DSMEM/TMA. Serves as a baseline before
introducing cluster + TMA multicast in the optimized variant.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError as exc:
    raise ImportError("Triton is required for this example") from exc


@triton.jit
def baseline_matmul_kernel(
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
    num_pid_m = tl.cdiv(M, BLOCK_M)
    pid_m = pid // num_pid_m
    pid_n = pid % num_pid_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_ptr = A + (offs_m[:, None] * stride_am + (k + offs_k)[None, :] * stride_ak)
        b_ptr = B + ((k + offs_k)[:, None] * stride_bk + offs_n[None, :] * stride_bn)
        a = tl.load(a_ptr, mask=offs_m[:, None] < M)
        b = tl.load(b_ptr, mask=offs_n[None, :] < N)
        acc += tl.dot(a, b)
    c_ptr = C + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptr, acc)


def run_baseline(M=1024, N=1024, K=1024, BLOCK=128):
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)

    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
    baseline_matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK, BLOCK, BLOCK,
    )
    return c


if __name__ == "__main__":
    torch.manual_seed(0)
    out = run_baseline()
    print(f"Baseline matmul completed, output mean={out.mean().item():.3f}")


# Minimal harness hook to allow skipping in automated sweeps.
try:
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
except Exception:
    BaseBenchmark = None  # type: ignore
    BenchmarkConfig = None  # type: ignore


def get_benchmark():
    if BaseBenchmark is None or BenchmarkConfig is None:
        return None
    import os
    class _SkipBenchmark(BaseBenchmark):
        def setup(self):  # pragma: no cover - skip in quick runs
            raise RuntimeError("SKIPPED: baseline_persistent_matmul_tma is manual-only in low-memory sweeps")
        def benchmark_fn(self): pass
        def get_config(self): return BenchmarkConfig(iterations=5, warmup=5)  # Minimum warmup required
    return _SkipBenchmark()

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )
