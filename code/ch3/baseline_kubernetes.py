"""Kubernetes baseline: per-iteration CPU orchestration with new tensors."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class BaselineKubernetesBenchmark(BaseBenchmark):
    """Allocates new tensors + launches multiple kernels every iteration."""

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
        ).to(self.device)
        # Two float32 batches per step: inputs + targets (512x1024 elements each)
        elements = 2 * 512 * 1024
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(elements),
            bytes_per_iteration=float(elements * 4),
        )

    def setup(self) -> None:
        torch.manual_seed(314)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
            bytes_per_iteration=self._workload.bytes_per_iteration,
        )

    def benchmark_fn(self) -> None:
        with self._nvtx_range("baseline_kubernetes"):
            host_data = torch.randn(512, 1024, dtype=torch.float32)
            host_target = torch.randn(512, 1024, dtype=torch.float32)
            data = host_data.to(self.device, non_blocking=False)
            target = host_target.to(self.device, non_blocking=False)
            self._synchronize()
            out = self.model(data)
            torch.nn.functional.mse_loss(out, target).backward()
            for p in self.model.parameters():
                p.grad = None
        self._synchronize()

    def teardown(self) -> None:
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 1),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineKubernetesBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline Kubernetes latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
