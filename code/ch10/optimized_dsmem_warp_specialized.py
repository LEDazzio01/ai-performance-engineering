"""Python harness wrapper for optimized_dsmem_warp_specialized.cu.

DSMEM Warp Specialized: Combines warp specialization with DSMEM for maximum throughput.

BOOK REFERENCE (Ch10): Warp specialization divides warps into different roles
for better resource utilization.

Key pattern:
  1. All warps perform block-level reduction with vectorized float4 loads
  2. Only warp 0 handles cluster communication via DSMEM
  3. Larger cluster (8 CTAs) for more DSMEM benefit

Optimizations:
  - Vectorized float4 loads for 4x bandwidth efficiency
  - Warp specialization reduces cross-CTA communication overhead
  - Dedicated communication warp avoids blocking compute warps
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


class OptimizedDSMEMWarpSpecializedBenchmark(CudaBinaryBenchmark):
    """Wraps DSMEM warp-specialized reduction."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_dsmem_warp_specialized",
            friendly_name="DSMEM Reduction (Warp Specialized + Vectorized)",
            iterations=3,
            warmup=5,
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
    return OptimizedDSMEMWarpSpecializedBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nDSMEM Warp Specialized: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

