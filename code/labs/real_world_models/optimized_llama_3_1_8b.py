#!/usr/bin/env python3
"""Optimized: Llama 3.1 8B with all Blackwell optimizations.

Enables torch.compile, FlexAttention, and optional FP8 for maximum performance.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, BenchmarkHarness, BenchmarkMode
from labs.real_world_models.llama_3_1_8b_optimization import Llama31_8B_Optimization


class OptimizedLlama31_8B(BaseBenchmark):
    """Optimized Llama 3.1 8B - torch.compile + FlexAttention + optional FP8.
    
    Optimizations:
    - torch.compile with max-autotune
    - FlexAttention for efficient attention
    - Optional FP8 precision (2x throughput on Blackwell)
    """

    def __init__(self, batch_size: int = 1, seq_length: int = 2048, use_fp8: bool = False):
        super().__init__()
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.use_fp8 = use_fp8
        self.model_wrapper = None
        self._last_metrics = {}

    def setup(self) -> None:
        self.model_wrapper = Llama31_8B_Optimization(
            batch_size=self.batch_size,
            seq_length=self.seq_length,
            use_compile=True,         # Enable torch.compile
            use_fp8=self.use_fp8,     # Optional FP8
            use_flex_attention=True,  # Enable FlexAttention
        )
        self.model_wrapper.setup()

    def benchmark_fn(self) -> None:
        if self.model_wrapper:
            self._last_metrics = self.model_wrapper.run()
        self._synchronize()

    def teardown(self) -> None:
        if self.model_wrapper:
            self.model_wrapper.teardown()
        self.model_wrapper = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=5)

    def get_custom_metrics(self) -> dict:
        metrics = self._last_metrics.copy()
        metrics["llama.use_compile"] = 1.0
        metrics["llama.use_flex_attention"] = 1.0
        metrics["llama.use_fp8"] = 1.0 if self.use_fp8 else 0.0
        return metrics


def get_benchmark() -> BaseBenchmark:
    return OptimizedLlama31_8B()


if __name__ == "__main__":
    from labs.real_world_models.baseline_llama_3_1_8b import BaselineLlama31_8B
    
    print("=== Llama 3.1 8B Optimization Comparison ===\n")
    
    # Run baseline
    baseline = BaselineLlama31_8B()
    harness = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=baseline.get_config())
    baseline_result = harness.benchmark(baseline)
    baseline_time = baseline_result.timing.mean_ms if baseline_result.timing else 0
    print(f"Baseline (eager): {baseline_time:.3f} ms")
    
    # Run optimized
    optimized = get_benchmark()
    harness2 = BenchmarkHarness(mode=BenchmarkMode.CUSTOM, config=optimized.get_config())
    optimized_result = harness2.benchmark(optimized)
    optimized_time = optimized_result.timing.mean_ms if optimized_result.timing else 0
    print(f"Optimized (compile+flex): {optimized_time:.3f} ms")
    
    if baseline_time > 0 and optimized_time > 0:
        speedup = baseline_time / optimized_time
        print(f"\nSpeedup: {speedup:.2f}x")
        print(f"Optimizations: torch.compile + FlexAttention")

