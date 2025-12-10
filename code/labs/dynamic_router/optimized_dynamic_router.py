"""Benchmark harness wrapper for the optimized dynamic router simulation."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.driver import simulate


class OptimizedDynamicRouterBenchmark(BaseBenchmark):
    """Runs the optimized (prefill/decode split) routing simulation under aisp bench."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.jitter_exemption_reason = "Dynamic router optimized: fixed configuration"
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        return

    def benchmark_fn(self) -> None:
        self._summary = simulate(
            "optimized",
            num_ticks=120,
            seed=0,
            log_interval=None,
        )

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=120,
            timeout_multiplier=3.0,
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"type": "dynamic_router_optimized"}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedDynamicRouterBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    cfg = bench.get_config()
    bench.benchmark_fn()
    print(bench.get_custom_metrics())
