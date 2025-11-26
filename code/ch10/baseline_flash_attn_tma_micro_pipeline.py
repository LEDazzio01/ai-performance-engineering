"""Baseline FlashAttention-style micro-pipeline with blocking copies."""

from __future__ import annotations
from typing import Optional

from pathlib import Path

from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode
from common.python.cuda_binary_benchmark import CudaBinaryBenchmark


class BaselineFlashAttnTmaMicroPipelineBenchmark(CudaBinaryBenchmark):
    """Runs the baseline flash-attn micro-pipeline (no async copies/TMA)."""

    def __init__(self) -> None:
        chapter_dir = Path(__file__).parent
        super().__init__(
            chapter_dir=chapter_dir,
            binary_name="baseline_flash_attn_tma_micro_pipeline",
            friendly_name="FlashAttn Micro-Pipeline Baseline (blocking copies)",
            iterations=1,
            warmup=0,
            timeout_seconds=120,
            run_args=(),
            # Require pipeline-capable/TMA GPUs to keep A/B runs aligned; SKIP otherwise.
            requires_pipeline_api=True,
        )


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_pipeline_metrics
        return compute_pipeline_metrics(
            num_stages=getattr(self, 'num_stages', 4),
            stage_times_ms=getattr(self, '_stage_times_ms', [1.0]),
        )

def get_benchmark() -> BaseBenchmark:
    return BaselineFlashAttnTmaMicroPipelineBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config(),
    )
    result = harness.benchmark(benchmark)
    mean_ms = result.timing.mean_ms if result.timing else 0.0
    print(f"\nBaseline flash-attn micro-pipeline time: {mean_ms:.3f} ms")
