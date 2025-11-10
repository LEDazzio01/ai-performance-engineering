"""Python harness wrapper for the CUDA baseline_matmul binary."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineMatmulCudaBenchmark(CudaBinaryBenchmark):
    """Wraps the naive batched matmul CUDA binary."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_matmul",
            friendly_name="Ch7 Baseline Matmul CUDA",
            iterations=3,
            warmup=1,
            timeout_seconds=180,
            time_regex=r"TIME_MS:\s*([0-9.]+)",
        )


def get_benchmark() -> BaselineMatmulCudaBenchmark:
    return BaselineMatmulCudaBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(
        f"\nCh7 Baseline Matmul CUDA: "
        f"{result.timing.mean_ms if result.timing else 0.0:.3f} ms"
    )
