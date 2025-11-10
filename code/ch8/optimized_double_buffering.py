"""Optimized double buffering benchmark with pipelined shared-memory tiles."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch8.double_buffering_benchmark_base import DoubleBufferingBenchmarkBase


class OptimizedDoubleBufferingBenchmark(DoubleBufferingBenchmarkBase):
    nvtx_label = "optimized_double_buffering"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.input is not None
        assert self.output is not None
        self.extension.double_buffer_optimized(self.input, self.output)


def get_benchmark() -> DoubleBufferingBenchmarkBase:
    return OptimizedDoubleBufferingBenchmark()


def main() -> None:
    from common.python.benchmark_harness import BenchmarkConfig, BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=30, warmup=5),
    )
    benchmark = OptimizedDoubleBufferingBenchmark()
    result = harness.benchmark(benchmark)
    print("=" * 70)
    print("Optimized Double Buffering")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()

