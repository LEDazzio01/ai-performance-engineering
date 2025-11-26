"""Loop-unrolling variant with ILP and vectorized loads."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch8.loop_unrolling_benchmark_base import LoopUnrollingBenchmarkBase


class OptimizedLoopUnrollingBenchmark(LoopUnrollingBenchmarkBase):
    nvtx_label = "optimized_loop_unrolling"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.inputs is not None
        assert self.weights is not None
        assert self.output is not None
        self.extension.loop_unrolling_optimized(self.inputs, self.weights, self.output)


    def get_custom_metrics(self) -> Optional[dict]:
        """Return optimization metrics for loop_unrolling."""
        from common.python.benchmark_metrics import compute_speedup_metrics
        return compute_speedup_metrics(
            baseline_ms=getattr(self, '_baseline_ms', 1.0),
            optimized_ms=getattr(self, '_last_elapsed_ms', 1.0),
            name="loop_unrolling",
        )

def get_benchmark() -> LoopUnrollingBenchmarkBase:
    return OptimizedLoopUnrollingBenchmark()


def main() -> None:
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=30, warmup=5),
    )
    benchmark = OptimizedLoopUnrollingBenchmark()
    result = harness.benchmark(benchmark)
    print("=" * 70)
    print("Optimized Loop Unrolling (ILP + vectorized loads)")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
