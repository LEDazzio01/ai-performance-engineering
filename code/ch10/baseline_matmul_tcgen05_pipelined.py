"""Baseline matmul benchmark: cuBLAS at larger matrix size.

CHAPTER 10 CONTEXT: This is the cuBLAS baseline for comparing against
the pipelined tcgen05 variant. Uses 8192x8192 matrices to better
demonstrate the gap between custom and vendor-optimized kernels.

WHY CUBLAS IS FAST:
1. Multi-stage software pipelining (3-4 stages typical)
2. Persistent kernel design for large matrices
3. Problem-specific tuning (different code paths per size)
4. Warp specialization for load vs compute
5. Optimal register blocking per architecture

The gap between basic tcgen05 and cuBLAS shows WHY these
optimizations matter for production performance.
"""

from __future__ import annotations

from typing import Optional

import torch

from ch10.optimized_matmul import resolve_device
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from common.python.tcgen05_requirements import check_tcgen05_support


class BaselineMatmulTCGen05PipelinedBenchmark(BaseBenchmark):
    """cuBLAS baseline at 8192x8192 for pipelined comparison."""

    def __init__(self) -> None:
        super().__init__()
        available, reason = check_tcgen05_support(
            loader=None,
            module_name="ch10 matmul tcgen05 kernels",
        )
        self._tcgen05_available = available
        self._skip_reason = reason or "SKIPPED: tcgen05 matmul unavailable"
        self.device = resolve_device()
        self.dtype = torch.float16
        # Match the pipelined variant size
        self.size = 8192
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        torch.manual_seed(0)
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=self.dtype)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        assert self.A is not None and self.B is not None
        with self._nvtx_range("baseline_matmul_cublas_8k"):
            with torch.no_grad():
                _ = torch.matmul(self.A, self.B)
        self._synchronize()

    def teardown(self) -> None:
        self.A = None
        self.B = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics."""
        flops = 2 * self.size ** 3
        return {
            "matrix_size": self.size,
            "theoretical_flops": flops,
            "library": "cuBLAS",
        }


def get_benchmark() -> BaselineMatmulTCGen05PipelinedBenchmark:
    return BaselineMatmulTCGen05PipelinedBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    print("cuBLAS Baseline (8192x8192)")
    print("=" * 50)
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    
    time_ms = result.timing.mean_ms if result.timing else 0.0
    size = benchmark.size
    flops = 2 * size ** 3
    tflops = (flops / 1e12) / (time_ms / 1000) if time_ms > 0 else 0
    
    print(f"Results ({size}x{size}x{size}):")
    print(f"  Time: {time_ms:.3f} ms")
    print(f"  Performance: {tflops:.1f} TFLOPS")

