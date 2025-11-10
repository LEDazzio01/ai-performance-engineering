"""baseline_cublas.py - Naive FP32 matmul without TF32 tensor-core acceleration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch2 cublas example")
    return torch.device("cuda")


class BaselineCublasBenchmark(Benchmark):
    """
    Baseline: FP32 matmul with TF32 disabled.

    Demonstrates the cost of ignoring tensor-core friendly settings before we
    introduce a pure cuBLAS/TF32 path in the optimized example.
    """

    def __init__(self):
        self.device = resolve_device()
        self.m = 2048
        self.n = 2048
        self.k = 2048
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self._prev_tf32_matmul: Optional[bool] = None
        self._prev_tf32_cudnn: Optional[bool] = None
        self._prev_precision: Optional[str] = None

    def setup(self) -> None:
        """Allocate FP32 matrices and disable TF32 acceleration."""
        self._prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        self._prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
        self._prev_precision = torch.get_float32_matmul_precision()

        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        torch.manual_seed(42)
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Plain cuBLAS FP32 matmul."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("baseline_cublas_fp32", enable=enable_nvtx):
            _ = torch.matmul(self.A, self.B)

    def teardown(self) -> None:
        """Restore TF32 settings and free tensors."""
        self.A = None
        self.B = None
        if self._prev_tf32_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = self._prev_tf32_matmul
        if self._prev_tf32_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = self._prev_tf32_cudnn
        if self._prev_precision is not None:
            torch.set_float32_matmul_precision(self._prev_precision)  # type: ignore[arg-type]
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineCublasBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5),
    )
    result = harness.benchmark(benchmark)
    timing = result.timing.mean_ms if result.timing else 0.0
    print(f"\nBaseline cuBLAS (FP32, TF32 disabled): {timing:.3f} ms")
