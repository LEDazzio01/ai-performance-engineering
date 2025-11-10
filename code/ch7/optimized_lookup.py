"""Python harness wrapper for ch7's optimized_lookup.cu."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedLookupBenchmark(CudaBinaryBenchmark):
    """Wraps the coalesced lookup kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_lookup",
            friendly_name="Ch7 Optimized Lookup",
            iterations=3,
            warmup=1,
            timeout_seconds=90,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
        )


def get_benchmark() -> OptimizedLookupBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedLookupBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nCh7 Optimized Lookup: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
