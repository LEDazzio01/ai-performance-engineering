"""Optimized vLLM dual-pool benchmark: dedicated prefill and decode pools."""

from __future__ import annotations

from typing import Dict, Optional

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.vllm_runner import run_dual_pool_vllm


class OptimizedDualPoolVllmBenchmark(BaseBenchmark):
    """Runs vLLM with disaggregated prefill and decode pools to cut TTFT tails."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}

    def setup(self) -> None:
        return

    def benchmark_fn(self) -> None:
        from labs.dynamic_router import vllm_runner

        self._summary = run_dual_pool_vllm("dual", cli_args=vllm_runner._CLI_ARGS)

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedDualPoolVllmBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    bench.benchmark_fn()
    print(bench.get_custom_metrics())
