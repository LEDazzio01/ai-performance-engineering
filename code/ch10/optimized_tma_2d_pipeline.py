"""Optimized wrapper for the Blackwell TMA 2D pipeline benchmark."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class OptimizedTma2DPipelineBenchmark(CudaBinaryBenchmark):
    """Runs the TMA-enabled pipeline to overlap cp.async tensor copies with compute."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="tma_2d_pipeline_blackwell",
            friendly_name="TMA 2D Pipeline Optimized (Tensor Memory Accelerator)",
            iterations=1,
            warmup=5,
            timeout_seconds=90,
            run_args=(),
            requires_pipeline_api=True,
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

def get_benchmark() -> CudaBinaryBenchmark:
    return OptimizedTma2DPipelineBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"\nOptimized TMA 2D pipeline time: {mean_ms:.3f} ms")
