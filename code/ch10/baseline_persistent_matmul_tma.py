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


# --- Benchmark Harness Integration ---
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig


class BaselinePersistentMatmulTMABenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Benchmark wrapper for baseline persistent matmul TMA."""

    def __init__(self, M: int = 4096, N: int = 4096, K: int = 4096):
        super().__init__()
        self.M = M
        self.N = N
        self.K = K
        self.result = None
        self.A = None
        self.B = None
        self.C = None
        self.block = 128
        self.output = None
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(M * N),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.cuda.empty_cache()
        self.A = torch.randn((self.M, self.K), device=self.device, dtype=torch.float16)
        self.B = torch.randn((self.K, self.N), device=self.device, dtype=torch.float16)
        self.C = torch.empty((self.M, self.N), device=self.device, dtype=torch.float16)
        grid = lambda META: (triton.cdiv(self.M, META["BLOCK_M"]) * triton.cdiv(self.N, META["BLOCK_N"]),)
        # One warmup to JIT the kernel and populate C
        baseline_matmul_kernel[grid](
            self.A, self.B, self.C,
            self.M, self.N, self.K,
            self.A.stride(0), self.A.stride(1),
            self.B.stride(0), self.B.stride(1),
            self.C.stride(0), self.C.stride(1),
            self.block, self.block, self.block,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        assert self.A is not None and self.B is not None and self.C is not None
        grid = lambda META: (triton.cdiv(self.M, META["BLOCK_M"]) * triton.cdiv(self.N, META["BLOCK_N"]),)
        baseline_matmul_kernel[grid](
            self.A, self.B, self.C,
            self.M, self.N, self.K,
            self.A.stride(0), self.A.stride(1),
            self.B.stride(0), self.B.stride(1),
            self.C.stride(0), self.C.stride(1),
            self.block, self.block, self.block,
        )
        self.result = self.C
        self._synchronize()
        self.output = self.result
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"A": self.A, "B": self.B},
            output=self.output.detach().float().clone(),
            batch_size=self.M,
            precision_flags={
                "fp16": True,
                "bf16": False,
                "fp8": False,
                "tf32": False,
            },
            output_tolerance=(1e-2, 1e-2),
        )

    def teardown(self) -> None:
        self.result = None
        self.A = None
        self.B = None
        self.C = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)


def get_benchmark() -> BaseBenchmark:
    return BaselinePersistentMatmulTMABenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
