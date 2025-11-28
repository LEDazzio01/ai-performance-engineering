"""Optimized occupancy tuning with higher ILP (unroll) at default block size."""

from __future__ import annotations

from pathlib import Path
import sys

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch8.baseline_occupancy_tuning import OccupancyBinaryBenchmark


class OptimizedOccupancyTuningBenchmark(OccupancyBinaryBenchmark):
    """Optimize occupancy: larger block (256), unroll (8), no heavy smem.
    
    Baseline artificially depresses occupancy with 45KB smem and small block.
    This optimized version removes that constraint and uses ILP via unrolling.
    """

    def __init__(self) -> None:
        super().__init__(
            friendly_name="Occupancy Tuning (block=256, unroll=8, no smem)",
            run_args=[
                "--block-size",
                "256",
                "--smem-bytes",
                "0",  # No heavy smem - allows higher occupancy
                "--unroll",
                "8",  # ILP via loop unrolling
                "--inner-iters",
                "1",  # Same work as baseline
                "--reps",
                "60",
            ],
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return optimization metrics for occupancy_tuning."""
        from common.python.benchmark_metrics import compute_speedup_metrics
        return compute_speedup_metrics(
            baseline_ms=getattr(self, '_baseline_ms', 1.0),
            optimized_ms=getattr(self, '_last_elapsed_ms', 1.0),
            name="occupancy_tuning",
        )

def get_benchmark() -> OptimizedOccupancyTuningBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedOccupancyTuningBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=benchmark.get_config())
    result = harness.benchmark(benchmark)
    print(f"\nOccupancy Tuning (maxrregcount=32): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
