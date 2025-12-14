"""Python harness wrapper for ch07's optimized_transpose_padded.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedTransposePaddedBenchmark(CudaBinaryBenchmark):
    """Wraps the padded tiled transpose kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        width = 4096
        matrix_bytes = width * width * 4
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_transpose_padded",
            friendly_name="Ch7 Optimized Transpose",
            iterations=3,
            warmup=5,
            timeout_seconds=90,
            workload_params={
                "width": width,
                "dtype": "float32",
            },
        )
        self.register_workload_metadata(bytes_per_iteration=float(matrix_bytes * 2))

    def get_custom_metrics(self) -> Optional[dict]:
        """Return memory access metrics."""
        return None

def get_benchmark() -> OptimizedTransposePaddedBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedTransposePaddedBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
