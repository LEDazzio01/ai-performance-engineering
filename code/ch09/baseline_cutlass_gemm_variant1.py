"""Python harness wrapper for baseline_cutlass_gemm_variant1.cu.

VARIANT 1: Fair comparison baseline with pre-loaded data
This ensures kernel execution time is measured without data transfer overhead.

BOOK REFERENCE (Ch9): When comparing GEMM implementations, it's critical to
isolate kernel execution from data movement to make fair comparisons.
"""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineCutlassGemmVariant1Benchmark(CudaBinaryBenchmark):
    """Wraps the fair comparison baseline CUTLASS GEMM (pre-loaded data)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_cutlass_gemm_variant1",
            friendly_name="Tiled GEMM Baseline (Fair - Preloaded)",
            iterations=5,
            warmup=5,
            timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline metrics for cutlass_gemm."""
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=self._total_flops,
            total_bytes=self._total_bytes,
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            precision="fp32",
        )

def get_benchmark() -> BaselineCutlassGemmVariant1Benchmark:
    """Factory for discover_benchmarks()."""
    return BaselineCutlassGemmVariant1Benchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nTiled GEMM Baseline (Variant1): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")


