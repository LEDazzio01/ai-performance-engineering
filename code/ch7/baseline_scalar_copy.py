"""Python harness wrapper for ch7's baseline_copy_scalar.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineCopyScalarBenchmark(CudaBinaryBenchmark):
    """Wraps the scalar copy kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_copy_scalar",
            friendly_name="Ch7 Scalar Copy",
            iterations=3,
            warmup=1,
            timeout_seconds=90,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics for scalar_copy."""
        from common.python.benchmark_metrics import compute_memory_access_metrics
        return compute_memory_access_metrics(
            bytes_requested=self._bytes_requested,
            bytes_actually_transferred=self._bytes_requested,  # Ideal case
            num_transactions=max(1, self._bytes_requested // 128),
            optimal_transactions=max(1, self._bytes_requested // 128),
        )

def get_benchmark() -> BaselineCopyScalarBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineCopyScalarBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nCh7 Scalar Copy (baseline): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

