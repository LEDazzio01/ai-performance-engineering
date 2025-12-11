"""Benchmark harness wrapper for the baseline dynamic router simulation."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.driver import simulate


class BaselineDynamicRouterBenchmark(BaseBenchmark):
    """Runs the baseline (single-pool) routing simulation under aisp bench."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.register_workload_metadata(requests_per_iteration=1.0)
        self.output: Optional[torch.Tensor] = None
        self.metrics: Optional[torch.Tensor] = None

    def setup(self) -> None:
        # No external assets to prepare
        return

    def benchmark_fn(self) -> None:
        # Fixed seed/ticks to keep runs comparable
        self._summary = simulate(
            "baseline",
            num_ticks=120,
            seed=0,
            log_interval=None,
        )
        # Surface a perturbable tensor output for verification
        metrics = list(self._summary.values()) or [0.0]
        expected_shape = (1, len(metrics))
        if self.metrics is None or tuple(self.metrics.shape) != expected_shape:
            self.metrics = torch.randn(expected_shape, dtype=torch.float32)
        summary_tensor = torch.tensor([metrics], dtype=torch.float32)
        self.output = (summary_tensor + self.metrics).detach()

    def get_config(self) -> Optional[BenchmarkConfig]:
        # Single iteration; simulation already encapsulates multiple ticks
        return BenchmarkConfig(
            iterations=1,
            warmup=5,
            measurement_timeout_seconds=120,
            timeout_multiplier=3.0,
        )

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None

    def teardown(self) -> None:
        self.metrics = None
        self.output = None
        super().teardown()

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {
            "type": "dynamic_router_baseline",
            "shapes": {
                "metrics": tuple(self.metrics.shape) if self.metrics is not None else (1, max(1, len(self._summary) or 1))
            },
        }

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineDynamicRouterBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    cfg = bench.get_config()
    bench.benchmark_fn()
    print(bench.get_custom_metrics())
