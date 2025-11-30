"""Occupancy tuning with block=64 and ILP for mid-range occupancy/ILP contrast."""

from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch08.baseline_occupancy_tuning import OccupancyBinaryBenchmark


class OptimizedOccupancyTuningBS64(OccupancyBinaryBenchmark):
    def __init__(self) -> None:
        super().__init__(
            friendly_name="Occupancy Tuning (block=64, unroll=8, inner=16)",
            run_args=["--block-size", "64", "--unroll", "8", "--inner-iters", "16", "--reps", "60"],
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return optimization metrics for occupancy_tuning_bs64."""
        from core.benchmark.metrics import compute_speedup_metrics
        return compute_speedup_metrics(
            baseline_ms=getattr(self, '_baseline_ms', 1.0),
            optimized_ms=getattr(self, '_last_elapsed_ms', 1.0),
            name="occupancy_tuning_bs64",
        )

def get_benchmark() -> OptimizedOccupancyTuningBS64:
    """Factory for discover_benchmarks()."""
    return OptimizedOccupancyTuningBS64()


if __name__ == "__main__":
    from core.harness.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(
        f"\nOccupancy Tuning (block=64, unroll=8, inner=12): {result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
