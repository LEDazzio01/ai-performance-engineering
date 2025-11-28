"""Baseline prefill/decode wrapper that skips when <2 GPUs are available."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig


class _SkipBenchmark(BaseBenchmark):
    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=1, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

    def benchmark_fn(self) -> None:
        raise RuntimeError("SKIPPED: prefill/decode multigpu requires >=2 GPUs")


def get_benchmark() -> BaseBenchmark:
    if torch.cuda.device_count() < 2:
        return _SkipBenchmark()
    from ch17.baseline_prefill_decode_disagg import BaselineInferenceMonolithicBenchmark
    return BaselineInferenceMonolithicBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BaseBenchmark, BenchmarkHarness, BenchmarkMode, BenchmarkConfig

    benchmark = get_benchmark()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=BenchmarkConfig(iterations=3, warmup=5))
    result = harness.benchmark(benchmark)
    print(f"Prefill/decode baseline (multi-GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
