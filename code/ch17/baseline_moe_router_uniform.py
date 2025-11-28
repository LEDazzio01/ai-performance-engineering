"""baseline_moe_router_uniform.py

Topology-agnostic MoE router: selects experts uniformly at random without
respecting NVLink/NVSwitch locality. Serves as a baseline for the topology-
aware router.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineMoERouterUniformBenchmark(BaseBenchmark):
    """Uniform expert selection ignoring fabric distance."""

    def __init__(self):
        super().__init__()
        self.num_experts = 16
        self.tokens = 4096
        self.experts: List[int] = list(range(self.num_experts))
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.tokens),
            tokens_per_iteration=float(self.tokens),
        )
        self._last_assignment: Dict[int, int] = {}

    def setup(self) -> None:
        torch.manual_seed(0)
        random.seed(0)
        self._synchronize()

    def benchmark_fn(self) -> None:
        with self._nvtx_range("baseline_moe_router_uniform"):
            assignment: Dict[int, int] = {}
            for token_id in range(self.tokens):
                expert = random.choice(self.experts)
                assignment[token_id] = expert
            self._last_assignment = assignment
            self._synchronize()

    def teardown(self) -> None:
        self._last_assignment = {}

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

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

    def validate_result(self) -> Optional[str]:
        if not self._last_assignment:
            return "No assignments produced"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineMoERouterUniformBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=2, warmup=5),
    )
    bench = BaselineMoERouterUniformBenchmark()
    result = harness.benchmark(bench)
    print(result)
