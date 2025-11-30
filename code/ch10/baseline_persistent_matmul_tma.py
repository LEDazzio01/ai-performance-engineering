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


# Benchmark harness hook for automated sweeps.
try:
    from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
except Exception:
    BaseBenchmark = None  # type: ignore
    BenchmarkConfig = None  # type: ignore


class BaselinePersistentMatmulTMABenchmark(BaseBenchmark):
    """Benchmark for baseline Triton matmul without TMA cluster multicast."""

    def __init__(self, M: int = 4096, N: int = 4096, K: int = 4096, BLOCK: int = 128):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.BLOCK = BLOCK
        self.a = None
        self.b = None
        self.c = None

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def setup(self) -> None:
        """Allocate input/output tensors on GPU."""
        self.a = torch.randn((self.M, self.K), device="cuda", dtype=torch.float16)
        self.b = torch.randn((self.K, self.N), device="cuda", dtype=torch.float16)
        self.c = torch.empty((self.M, self.N), device="cuda", dtype=torch.float16)

    def benchmark_fn(self) -> None:
        """Run the baseline Triton matmul kernel."""
        grid = lambda META: (triton.cdiv(self.M, META["BLOCK_M"]) * triton.cdiv(self.N, META["BLOCK_N"]),)
        baseline_matmul_kernel[grid](
            self.a, self.b, self.c,
            self.M, self.N, self.K,
            self.a.stride(0), self.a.stride(1),
            self.b.stride(0), self.b.stride(1),
            self.c.stride(0), self.c.stride(1),
            self.BLOCK, self.BLOCK, self.BLOCK,
        )

    def get_custom_metrics(self) -> dict:
        """Return domain-specific metrics for matmul."""
        from core.benchmark.metrics import compute_matmul_metrics
        return compute_matmul_metrics(
            M=self.M, N=self.N, K=self.K,
            dtype_bytes=2,  # float16
        )


def get_benchmark():
    if BaseBenchmark is None or BenchmarkConfig is None:
        return None
    if not torch.cuda.is_available():
        class _SkipBenchmark(BaseBenchmark):
            def setup(self):
                raise RuntimeError("SKIPPED: CUDA required for baseline_persistent_matmul_tma")
            def benchmark_fn(self): pass
            def get_config(self): return BenchmarkConfig(iterations=1, warmup=5)
        return _SkipBenchmark()
    return BaselinePersistentMatmulTMABenchmark()
