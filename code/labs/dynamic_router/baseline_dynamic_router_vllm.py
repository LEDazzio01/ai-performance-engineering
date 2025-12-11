"""Baseline vLLM-backed routing benchmark (no feedback control)."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.vllm_runner import run_vllm_routing


class BaselineDynamicRouterVllmBenchmark(BaseBenchmark):
    """Runs vLLM without feedback routing (round-robin placement)."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.metrics: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        return

    def benchmark_fn(self) -> None:
        from labs.dynamic_router import vllm_runner

        self._summary = run_vllm_routing("baseline", cli_args=vllm_runner._CLI_ARGS)
        metric_values = list(self._summary.values()) or [0.0]
        expected_shape = (1, len(metric_values))
        if self.metrics is None or tuple(self.metrics.shape) != expected_shape:
            self.metrics = torch.randn(expected_shape, dtype=torch.float32)
        summary_tensor = torch.tensor([metric_values], dtype=torch.float32)
        self.output = (summary_tensor + self.metrics).detach()

    def teardown(self) -> None:
        self.metrics = None
        self.output = None
        super().teardown()

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        if self.output is None:
            raise RuntimeError("benchmark_fn() must be called before verification")
        return self.output

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        shape = tuple(self.metrics.shape) if self.metrics is not None else (1, max(1, len(self._summary) or 1))
        return {
            "type": "dynamic_router_vllm_baseline",
            "shapes": {"metrics": shape},
        }

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineDynamicRouterVllmBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    bench.benchmark_fn()
    print(bench.get_custom_metrics())
