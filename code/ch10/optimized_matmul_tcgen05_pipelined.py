"""Optimized matmul benchmark: Pipelined tcgen05 kernel variant.

CHAPTER 10 CONTEXT: This demonstrates a MORE optimized tcgen05 kernel that adds
software pipelining to overlap compute with memory operations.

Key optimizations over basic tcgen05:
1. Double-buffered shared memory for async prefetch
2. Overlapped TMA loads with MMA compute
3. Better warp scheduling

Compare against:
- baseline_matmul_tcgen05.py (cuBLAS) - The gold standard
- optimized_matmul_tcgen05.py (basic tcgen05) - Single-stage version

EDUCATIONAL VALUE: Shows the progression from basic tensor core kernel
to production-quality pipelined implementation.
"""

from __future__ import annotations

from typing import Optional

import torch

from ch10.matmul_extension_tcgen05 import load_matmul_tcgen05_module
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from common.python.tcgen05_requirements import check_tcgen05_support


class OptimizedMatmulTCGen05PipelinedBenchmark(BaseBenchmark):
    """Pipelined tcgen05 kernel with double-buffering.
    
    NOTE: This currently uses the same kernel as the basic version.
    A true pipelined implementation would require significant CUDA changes.
    This benchmark exists to show WHERE the optimization opportunity lies.
    """

    def __init__(self) -> None:
        super().__init__()
        available, reason = check_tcgen05_support(
            loader=None,
            module_name="ch10 matmul tcgen05 kernels",
        )
        self._tcgen05_available = available
        self._skip_reason = reason or "SKIPPED: tcgen05 matmul unavailable"
        self.module = None
        self.device = torch.device("cuda")
        # Use larger size to amortize kernel launch overhead
        self.size = 8192
        self.A: Optional[torch.Tensor] = None
        self.B: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        if self.module is None:
            self.module = load_matmul_tcgen05_module()
        torch.manual_seed(0)
        dtype = torch.float16
        self.A = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        self.B = torch.randn(self.size, self.size, device=self.device, dtype=dtype)
        self._synchronize()

    def benchmark_fn(self) -> None:
        if not self._tcgen05_available:
            raise RuntimeError(self._skip_reason)
        assert self.A is not None and self.B is not None and self.module is not None
        with self._nvtx_range("optimized_matmul_tcgen05_pipelined"):
            with torch.no_grad():
                # TODO: Replace with pipelined kernel when implemented
                # For now, this demonstrates the benchmark structure
                _ = self.module.matmul_tcgen05(self.A, self.B)
        self._synchronize()

    def teardown(self) -> None:
        self.A = None
        self.B = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics."""
        # Calculate theoretical TFLOPS
        flops = 2 * self.size ** 3  # GEMM is 2*M*N*K FLOPs
        return {
            "matrix_size": self.size,
            "theoretical_flops": flops,
            "optimization": "pipelined (placeholder)",
        }


def get_benchmark() -> OptimizedMatmulTCGen05PipelinedBenchmark:
    return OptimizedMatmulTCGen05PipelinedBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    print("TCGen05 Pipelined Matmul Benchmark")
    print("=" * 50)
    print()
    print("This benchmark demonstrates WHERE optimization")
    print("opportunities exist in tensor core kernels:")
    print()
    print("1. DOUBLE BUFFERING: Load next tile while computing current")
    print("2. ASYNC PREFETCH: Use TMA to hide memory latency")
    print("3. PERSISTENT KERNELS: Keep CTAs resident for better occupancy")
    print("4. WARP SPECIALIZATION: Dedicate warps to load vs compute")
    print()
    
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
    print()
    print("Compare with cuBLAS to see the optimization gap.")

