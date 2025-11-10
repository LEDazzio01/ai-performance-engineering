"""Baseline matmul benchmark used for tcgen05 comparisons."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch10.matmul_extension_tcgen05 import load_matmul_tcgen05_module
from ch10.optimized_matmul import resolve_device
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from common.python.tcgen05_requirements import check_tcgen05_support


class BaselineMatmulTCGen05Benchmark(Benchmark):
    """Uses PyTorch tensor core matmul as the baseline for tcgen05."""

    def __init__(self) -> None:
        available, reason = check_tcgen05_support(
            loader=None,
            module_name="ch10 matmul tcgen05 kernels",
        )
        self._tcgen05_available = available
        self._skip_reason = reason or "SKIPPED: tcgen05 matmul unavailable"
        self.device = resolve_device()
        self.dtype = torch.float16
        self.size = 4096
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        torch.manual_seed(0)
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)

    def benchmark_fn(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        assert self.A is not None and self.B is not None
        with torch.no_grad():
            _ = torch.matmul(self.A, self.B)

    def teardown(self) -> None:
        self.A = None
        self.B = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def validate_result(self) -> Optional[str]:
        if not self._tcgen05_available:
            return self._skip_reason
        if self.A is None or self.B is None:
            return "Matrices not initialized"
        return None


def get_benchmark() -> BaselineMatmulTCGen05Benchmark:
    return BaselineMatmulTCGen05Benchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(
        f"\nBaseline tcgen05 matmul (PyTorch tensor cores): "
        f"{result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
