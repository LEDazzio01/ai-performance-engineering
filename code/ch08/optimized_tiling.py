"""Optimized tiling benchmark that reuses shared-memory tiles."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from ch08.tiling_benchmark_base import TilingBenchmarkBase


class OptimizedTilingBenchmark(TilingBenchmarkBase):
    """Optimized implementation that loads tiles into shared memory."""

    nvtx_label = "optimized_tiling"

    def _invoke_kernel(self) -> None:
        assert self.extension is not None
        assert self.matrix_a is not None
        assert self.matrix_b is not None
        assert self.output is not None
        if hasattr(self.extension, "matmul_tiled_fast"):
            self.extension.matmul_tiled_fast(self.matrix_a, self.matrix_b, self.output)
        else:
            self.extension.matmul_tiled(self.matrix_a, self.matrix_b, self.output)


    def get_custom_metrics(self) -> Optional[dict]:
        """Return optimization metrics for tiling."""
        from core.benchmark.metrics import compute_speedup_metrics
        return compute_speedup_metrics(
            baseline_ms=getattr(self, '_baseline_ms', 1.0),
            optimized_ms=getattr(self, '_last_elapsed_ms', 1.0),
            name="tiling",
        )

def get_benchmark() -> TilingBenchmarkBase:
    """Factory function for harness discovery."""
    return OptimizedTilingBenchmark()


def main() -> None:
    """Allow `python optimized_tiling.py` for manual profiling."""
    from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5),
    )
    benchmark = OptimizedTilingBenchmark()
    result = harness.benchmark(benchmark)

    print("=" * 70)
    print("Optimized: Tiling (shared-memory reuse)")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
