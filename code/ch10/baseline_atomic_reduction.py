"""Python harness wrapper for baseline_atomic_reduction.cu.

Chapter 10 - DSMEM-Free Baseline
This is the two-pass block reduction approach that works on ANY CUDA device.

Compare with optimized_atomic_reduction.py (single-pass atomic).
"""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineAtomicReductionBenchmark(CudaBinaryBenchmark):
    """Wraps the two-pass block reduction kernel (DSMEM-free baseline)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_atomic_reduction",
            friendly_name="Two-Pass Block Reduction (DSMEM-Free)",
            iterations=3,
            warmup=5,
            timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics."""
        from common.python.benchmark_metrics import compute_bandwidth_metrics
        return compute_bandwidth_metrics(
            total_bytes=getattr(self, '_total_bytes', 64 * 1024 * 1024 * 4),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
        )


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineAtomicReductionBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nTwo-Pass Reduction: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")


