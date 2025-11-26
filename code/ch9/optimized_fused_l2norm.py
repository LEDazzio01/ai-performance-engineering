"""Python harness wrapper for optimized_fused_l2norm.cu."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedFusedL2NormBenchmark(CudaBinaryBenchmark):
    """Wraps the optimized fused L2 norm kernel."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_fused_l2norm",
            friendly_name="Optimized Fused L2 Norm",
            iterations=5,
            warmup=1,
            timeout_seconds=90,
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline metrics for fused_l2norm."""
        from common.python.benchmark_metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=self._total_flops,
            total_bytes=self._total_bytes,
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            precision="fp16",
        )

def get_benchmark() -> OptimizedFusedL2NormBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedFusedL2NormBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Fused L2 Norm: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

