"""Loop-unrolling baseline that keeps redundant inner loops."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch8.loop_unrolling_benchmark_base import LoopUnrollingBenchmarkBase


class BaselineLoopUnrollingBenchmark(LoopUnrollingBenchmarkBase):
    nvtx_label = "baseline_loop_unrolling"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.inputs is not None
        assert self.weights is not None
        assert self.output is not None
        self.extension.loop_unrolling_baseline(self.inputs, self.weights, self.output)


def get_benchmark() -> LoopUnrollingBenchmarkBase:
    return BaselineLoopUnrollingBenchmark()


def main() -> None:
    from common.python.benchmark_harness import BenchmarkConfig, BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=30, warmup=5),
    )
    benchmark = BaselineLoopUnrollingBenchmark()
    result = harness.benchmark(benchmark)
    print("=" * 70)
    print("Baseline Loop Unrolling (redundant inner loop)")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
