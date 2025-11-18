"""Optimized Pallas TMEM GEMM (double buffered) wired into the benchmark harness."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch9.pallas_tmem_common import PallasTmemBenchmark, PallasTmemConfig


class OptimizedPallasTmemBenchmark(PallasTmemBenchmark):
    """Optimized: double-buffered TMEM pipeline and epilogue reuse."""

    def __init__(self) -> None:
        super().__init__(
            PallasTmemConfig(max_concurrent_steps=2),
            friendly_name="optimized_tmem_pallas",
        )


def get_benchmark() -> OptimizedPallasTmemBenchmark:
    return OptimizedPallasTmemBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(
        f"\nOptimized TMEM Pallas: {result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
