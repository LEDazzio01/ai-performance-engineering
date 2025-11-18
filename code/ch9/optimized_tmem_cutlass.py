"""Optimized TMEM CUTLASS wrapper for harness discovery.

This re-exports the Chapter 10 tcgen05 optimized kernel (CUTLASS/CuTe + TMEM
epilogue) under the `tmem_cutlass` example name so discovery pairs it with the
baseline in this chapter.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch10.optimized_matmul_tcgen05 import OptimizedMatmulTCGen05Benchmark


class OptimizedTmemCutlassBenchmark(OptimizedMatmulTCGen05Benchmark):
    """CUTLASS/CuTe TMEM benchmark alias for Chapter 9."""


def get_benchmark() -> OptimizedTmemCutlassBenchmark:
    return OptimizedTmemCutlassBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(
        f"\nOptimized TMEM CUTLASS: {result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
