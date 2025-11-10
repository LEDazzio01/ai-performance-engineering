"""Ch7 optimized memory access benchmark (coalesced float4 copy)."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedMemoryAccessBenchmark(CudaBinaryBenchmark):
    """Wraps the coalesced CUDA kernel optimized implementation."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_memory_access",
            friendly_name="Ch7 Optimized Memory Access",
            iterations=3,
            warmup=1,
            timeout_seconds=90,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
        )


def get_benchmark() -> OptimizedMemoryAccessBenchmark:
    return OptimizedMemoryAccessBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(
        f"\nCh7 Optimized Memory Access: "
        f"{result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
