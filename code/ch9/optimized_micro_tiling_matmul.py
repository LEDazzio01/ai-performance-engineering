"""Python harness wrapper for optimized_micro_tiling_matmul.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedMicroTilingMatmulBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized micro-tiling matmul kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_micro_tiling_matmul",
            friendly_name="Optimized Micro-tiling Matmul",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline metrics for micro_tiling_matmul."""
        from common.python.benchmark_metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=self._total_flops,
            total_bytes=self._total_bytes,
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            precision="fp16",
        )

def get_benchmark() -> OptimizedMicroTilingMatmulBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedMicroTilingMatmulBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Micro-tiling Matmul: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

