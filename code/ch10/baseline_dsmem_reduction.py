"""Python harness wrapper for baseline_dsmem_reduction.cu - Two-Pass Reduction Baseline."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark
class BaselineDSMEMReductionBenchmark(CudaBinaryBenchmark):
    """Wraps the two-pass reduction kernel (baseline for DSMEM comparison)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_dsmem_reduction",
            friendly_name="Two-Pass Reduction (Baseline)",
            iterations=3,
            warmup=1,
            timeout_seconds=120,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )
def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineDSMEMReductionBenchmark()
if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nTwo-Pass Reduction: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

