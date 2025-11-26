"""Python harness wrapper for optimized_tma_multicast.cu - TMA Multicast for Clusters."""

from __future__ import annotations
from typing import Optional

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark
from common.python.hardware_capabilities import ensure_dsmem_supported
class _SkipBenchmark(BaseBenchmark):
    """Placeholder when TMA multicast can't run on this hardware."""
    def benchmark_fn(self) -> None:
        raise RuntimeError("SKIPPED: TMA multicast requires SM 9.0+ (Hopper/Blackwell)")
class OptimizedTMAMulticastBenchmark(CudaBinaryBenchmark):
    """Wraps the TMA multicast GEMM kernel demonstrating cluster-wide broadcasts."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="optimized_tma_multicast",
            friendly_name="TMA Multicast GEMM",
            iterations=3,
            warmup=1,
            timeout_seconds=180,
        )

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )
def _is_sm90_or_higher() -> bool:
    """Check if running on SM 9.0+ where TMA multicast is available."""
    if not torch.cuda.is_available():
        return False
    major, _ = torch.cuda.get_device_capability()
    return major >= 9
def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    if not _is_sm90_or_higher():
        return _SkipBenchmark()
    return OptimizedTMAMulticastBenchmark()
if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    print(f"\nTMA Multicast GEMM: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")

