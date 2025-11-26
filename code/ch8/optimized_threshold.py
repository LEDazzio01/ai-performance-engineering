"""Predicated threshold benchmark."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch8.threshold_benchmark_base import ThresholdBenchmarkBase


class OptimizedThresholdBenchmark(ThresholdBenchmarkBase):
    nvtx_label = "optimized_threshold"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.inputs is not None
        assert self.outputs is not None
        self.extension.threshold_optimized(self.inputs, self.outputs, self.threshold)


    def get_custom_metrics(self) -> Optional[dict]:
        """Return optimization metrics for threshold."""
        from common.python.benchmark_metrics import compute_speedup_metrics
        return compute_speedup_metrics(
            baseline_ms=getattr(self, '_baseline_ms', 1.0),
            optimized_ms=getattr(self, '_last_elapsed_ms', 1.0),
            name="threshold",
        )

def get_benchmark() -> ThresholdBenchmarkBase:
    return OptimizedThresholdBenchmark()


def main() -> None:
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5),
    )
    benchmark = OptimizedThresholdBenchmark()
    result = harness.benchmark(benchmark)
    print("=" * 70)
    print("Optimized Threshold (predicated)")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()

