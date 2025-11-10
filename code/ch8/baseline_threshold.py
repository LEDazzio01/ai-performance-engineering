"""Branch-heavy threshold benchmark baseline."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch8.threshold_benchmark_base import ThresholdBenchmarkBase


class BaselineThresholdBenchmark(ThresholdBenchmarkBase):
    nvtx_label = "baseline_threshold"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.host_inputs is not None
        assert self.inputs is not None
        assert self.outputs is not None
        # Naive implementation streams data from host memory each iteration before running on the GPU.
        self.inputs.copy_(self.host_inputs, non_blocking=False)
        self.extension.threshold_baseline(self.inputs, self.outputs, self.threshold)


def get_benchmark() -> ThresholdBenchmarkBase:
    return BaselineThresholdBenchmark()


def main() -> None:
    from common.python.benchmark_harness import BenchmarkConfig, BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5),
    )
    benchmark = BaselineThresholdBenchmark()
    result = harness.benchmark(benchmark)
    print("=" * 70)
    print("Baseline Threshold (branch divergence)")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
