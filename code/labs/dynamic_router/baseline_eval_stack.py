"""Baseline cheap eval stack: minimal telemetry and routing heuristics."""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.eval_stack import EvalConfig, run_eval_stack


class BaselineEvalStackBenchmark(BaseBenchmark):
    """Runs the mock cheap-eval stack with baseline settings."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.metrics: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.register_workload_metadata(requests_per_iteration=1.0)

    def _resolve_device(self) -> torch.device:  # type: ignore[override]
        return torch.device("cpu")

    def setup(self) -> None:
        return

    def benchmark_fn(self) -> None:
        cfg = EvalConfig.from_flags(self._argv(), seed=0)
        self._summary = run_eval_stack("baseline", cfg)
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

    def _argv(self) -> List[str]:
        """Pull target-specific extra args from the harness config if available."""
        cfg = getattr(self, "_config", None)
        if cfg is None:
            return sys.argv[1:]
        label = getattr(cfg, "target_label", None)
        extra_map = getattr(cfg, "target_extra_args", {}) or {}
        if label and label in extra_map:
            return list(extra_map[label])
        if len(extra_map) == 1:
            return list(next(iter(extra_map.values())))
        return sys.argv[1:]

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5, measurement_timeout_seconds=90)

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
        return {"type": "eval_stack_baseline", "shapes": {"metrics": shape}}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineEvalStackBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    bench.benchmark_fn()
    print(bench.get_custom_metrics())
