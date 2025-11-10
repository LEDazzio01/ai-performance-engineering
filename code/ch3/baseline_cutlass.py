"""Baseline GEMM that disables tensor-core friendly settings (no CUTLASS)."""

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
        raise RuntimeError("CUDA required for ch3")
    return torch.device("cuda")


class BaselineCutlassBenchmark(Benchmark):
    """FP32 GEMM with TF32 disabled and no kernel warmup."""

    def __init__(self):
        self.device = resolve_device()
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None
        self._orig_tf32_matmul: Optional[bool] = None
        self._orig_tf32_cudnn: Optional[bool] = None
        self.m = 3072
        self.n = 3072
        self.k = 3072

    def setup(self) -> None:
        torch.manual_seed(42)
        self._orig_tf32_matmul = torch.backends.cuda.matmul.allow_tf32
        self._orig_tf32_cudnn = torch.backends.cudnn.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        self.A = torch.randn(self.m, self.k, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.k, self.n, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.A is not None and self.B is not None
        with nvtx_range("baseline_cutlass", enable=enable_nvtx):
            prod = torch.matmul(self.A, self.B)
            _ = prod.relu_()  # extra epilogue w/o fusion

    def teardown(self) -> None:
        self.A = None
        self.B = None
        if self._orig_tf32_matmul is not None:
            torch.backends.cuda.matmul.allow_tf32 = self._orig_tf32_matmul
        if self._orig_tf32_cudnn is not None:
            torch.backends.cudnn.allow_tf32 = self._orig_tf32_cudnn
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=4)

    def validate_result(self) -> Optional[str]:
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineCutlassBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline CUTLASS latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
