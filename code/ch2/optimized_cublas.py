"""optimized_cublas.py - Pure cuBLAS matmul with TF32 tensor-core acceleration."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch2 cublas example")
    return torch.device("cuda")


class OptimizedCublasBenchmark(Benchmark):
    """
    Optimized: pure cuBLAS GEMM with TF32 and warmed-up heuristics.

    This keeps the math in FP32 but lets cuBLAS route workloads through tensor cores
    (TF32) while running a few warmup matmuls so Lt heuristics cache the best kernel.
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
        """Enable TF32, allocate FP32 matrices, and warm up cuBLAS."""
        self._prev_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        self._prev_tf32_cudnn = torch.backends.cudnn.allow_tf32
        self._prev_precision = torch.get_float32_matmul_precision()

        enable_tf32()
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

        torch.manual_seed(42)
        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

        # Warmup a handful of GEMMs so cuBLAS Lt heuristics settle before measurement.
        for _ in range(10):
            _ = torch.matmul(self.A, self.B)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """cuBLAS TF32 GEMM."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        with nvtx_range("cublas", enable=enable_nvtx):
            _ = torch.matmul(self.A, self.B)

    def teardown(self) -> None:
        """Restore TF32 knobs and free tensors."""
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
    return OptimizedCublasBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5),
    )
    result = harness.benchmark(benchmark)
    timing = result.timing.mean_ms if result.timing else 0.0
    print(f"\nOptimized cuBLAS (TF32): {timing:.3f} ms")
