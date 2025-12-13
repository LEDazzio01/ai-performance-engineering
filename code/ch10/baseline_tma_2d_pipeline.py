"""Baseline wrapper for the TMA 2D pipeline without TMA descriptors."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from core.benchmark.cuda_binary_benchmark import CudaBinaryBenchmark
from core.benchmark.verification import simple_signature


class BaselineTma2DPipelineBenchmark(CudaBinaryBenchmark):
    """Runs tma_2d_pipeline_blackwell.cu with --force-fallback to disable TMA."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="tma_2d_pipeline_blackwell",
            friendly_name="TMA 2D Pipeline Baseline (Fallback Copies)",
            iterations=1,
            warmup=1,
            timeout_seconds=90,
            run_args=("--baseline-only",),
            # Fallback path does not require CUDA pipeline APIs.
            requires_pipeline_api=False,
            workload_params={
                "batch_size": 4096,
                "dtype": "float32",
                "M": 4096,
                "N": 4096,
            },
        )
        self.register_workload_metadata(bytes_per_iteration=1024 * 1024)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

    def get_input_signature(self) -> dict:
        """Signature for baseline TMA 2D pipeline (fallback copies)."""
        return simple_signature(
            batch_size=4096,
            dtype="float32",
            M=4096,
            N=4096,
        ).to_dict()

    def get_output_tolerance(self) -> tuple[float, float]:
        return (0.0, 0.0)

def get_benchmark() -> CudaBinaryBenchmark:
    return BaselineTma2DPipelineBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
