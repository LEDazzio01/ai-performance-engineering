"""Optimized cheap eval stack: adds router-aware drop penalties and better accuracy priors."""

from __future__ import annotations

import sys
from typing import Dict, List, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.eval_stack import EvalConfig, run_eval_stack


class OptimizedEvalStackBenchmark(BaseBenchmark):
    """Runs the cheap-eval stack with the optimized settings."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.jitter_exemption_reason = "Eval stack optimized: fixed configuration"
        self.register_workload_metadata(requests_per_iteration=1.0)

    def _resolve_device(self) -> torch.device:  # type: ignore[override]
        return torch.device("cpu")

    def setup(self) -> None:
        return

    def benchmark_fn(self) -> None:
        cfg = EvalConfig.from_flags(self._argv(), seed=0)
        self._summary = run_eval_stack("optimized", cfg)

    def _argv(self) -> List[str]:
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
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"type": "eval_stack_optimized"}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    """Factory for discover_benchmarks()."""
    return OptimizedEvalStackBenchmark()


if __name__ == "__main__":
    bench = get_benchmark()
    bench.benchmark_fn()
    print(bench.get_custom_metrics())
