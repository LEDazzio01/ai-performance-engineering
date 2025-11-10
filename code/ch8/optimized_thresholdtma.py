"""Optimized threshold benchmark using CUDA pipeline/TMA staging."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch8.threshold_tma_benchmark_base import ThresholdBenchmarkBaseTMA


class OptimizedThresholdTMABenchmark(ThresholdBenchmarkBaseTMA):
    nvtx_label = "optimized_threshold_tma"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.inputs is not None
        assert self.outputs is not None
        self.extension.threshold_tma_optimized(self.inputs, self.outputs, self.threshold)


def get_benchmark() -> ThresholdBenchmarkBaseTMA:
    return OptimizedThresholdTMABenchmark()


def main() -> None:
    from common.python.benchmark_harness import BenchmarkConfig, BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5),
    )
    benchmark = OptimizedThresholdTMABenchmark()
    result = harness.benchmark(benchmark)
    print("=" * 70)
    print("Optimized Threshold (Blackwell TMA pipeline)")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
