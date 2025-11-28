"""Baseline cheap eval stack: minimal telemetry and routing heuristics."""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.eval_stack import EvalConfig, run_eval_stack


class BaselineEvalStackBenchmark(BaseBenchmark):
    """Runs the mock cheap-eval stack with baseline settings."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}

    def _resolve_device(self) -> torch.device:  # type: ignore[override]
        return torch.device("cpu")

    def setup(self) -> None:
        return

    def benchmark_fn(self) -> None:
        cfg = EvalConfig.from_flags(self._argv(), seed=0)
        self._summary = run_eval_stack("baseline", cfg)

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


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return BaselineEvalStackBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    bench.benchmark_fn()
    print(bench.get_custom_metrics())
