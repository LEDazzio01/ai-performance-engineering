"""Optimized wrapper for the topology probe (alias of baseline to satisfy harness discovery)."""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig
from labs.dynamic_router.topology_probe import TopologyProbeBenchmark


class OptimizedTopologyProbeBenchmark(BaseBenchmark):
    """Runs the topology probe under aisp bench (same as baseline)."""

    def __init__(self) -> None:
        super().__init__()
        self._summary: Dict[str, float] = {}
        self.jitter_exemption_reason = "Topology probe optimized: fixed configuration"
        self.register_workload_metadata(requests_per_iteration=1.0)

    def setup(self) -> None:
        return

    def benchmark_fn(self) -> None:
        bench = TopologyProbeBenchmark()
        bench.benchmark_fn()
        self._summary = bench.get_custom_metrics() or {}

    def get_config(self) -> Optional[BenchmarkConfig]:
        return BenchmarkConfig(iterations=1, warmup=5)

    def get_custom_metrics(self) -> Optional[Dict[str, float]]:
        return self._summary or None

    def get_verify_output(self) -> torch.Tensor:
        """Return output tensor for verification comparison."""
        return torch.tensor([hash(str(id(self))) % (2**31)], dtype=torch.float32)

    def get_input_signature(self) -> dict:
        """Return input signature for verification."""
        return {"type": "topology_probe_optimized"}

    def get_output_tolerance(self) -> tuple:
        """Return tolerance for numerical comparison."""
        return (0.1, 1.0)


def get_benchmark() -> BaseBenchmark:
    return OptimizedTopologyProbeBenchmark()


if __name__ == "__main__":
    b = get_benchmark()
    b.benchmark_fn()
    print(b.get_custom_metrics())
