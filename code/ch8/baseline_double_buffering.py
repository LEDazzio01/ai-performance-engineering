"""Baseline double buffering benchmark without pipelining."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch8.double_buffering_benchmark_base import DoubleBufferingBenchmarkBase


class BaselineDoubleBufferingBenchmark(DoubleBufferingBenchmarkBase):
    nvtx_label = "baseline_double_buffering"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.host_input is not None
        assert self.input is not None
        assert self.output is not None
        # Naive path: stage data from host memory into device buffer every iteration.
        self.input.copy_(self.host_input, non_blocking=False)
        self.extension.double_buffer_baseline(self.input, self.output)


def get_benchmark() -> DoubleBufferingBenchmarkBase:
    return BaselineDoubleBufferingBenchmark()


def main() -> None:
    from common.python.benchmark_harness import BenchmarkConfig, BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=30, warmup=5),
    )
    benchmark = BaselineDoubleBufferingBenchmark()
    result = harness.benchmark(benchmark)
    print("=" * 70)
    print("Baseline Double Buffering")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
