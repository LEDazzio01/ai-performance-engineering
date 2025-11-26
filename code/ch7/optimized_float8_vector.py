"""Python harness wrapper for optimized_float8_vector.cu - 32-byte Vectorized Loads."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark
class OptimizedFloat8VectorBenchmark(CudaBinaryBenchmark):
    """Wraps the 32-byte vectorized load benchmark for Blackwell."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_float8_vector",
            friendly_name="32-byte Vectorized Loads",
            iterations=3,
            warmup=1,
            timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics for float8_vector."""
        from common.python.benchmark_metrics import compute_memory_access_metrics
        return compute_memory_access_metrics(
            bytes_requested=self._bytes_requested,
            bytes_actually_transferred=self._bytes_requested,  # Ideal case
            num_transactions=max(1, self._bytes_requested // 128),
            optimal_transactions=max(1, self._bytes_requested // 128),
        )
def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedFloat8VectorBenchmark()
if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\n32-byte Vectorized Loads: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

