"""NUMA-unaware baseline: copies pageable CPU tensors to GPU each step."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class BaselineNUMAUnawareBenchmark(BaseBenchmark):
    """Allocates pageable host memory and blocks on every copy."""

    def __init__(self):
        super().__init__()
        self.host_tensor: Optional[torch.Tensor] = None
        self.device_buffer: Optional[torch.Tensor] = None
        bytes_per_iter = 128_000_000 * 4  # float32 bytes
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter),
        )

    def setup(self) -> None:
        torch.manual_seed(9)
        self.host_tensor = torch.randn(128_000_000, dtype=torch.float32)  # ~512 MB
        self.device_buffer = torch.empty_like(self.host_tensor, device=self.device)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            bytes_per_iteration=self._workload.bytes_per_iteration,
        )

    def benchmark_fn(self) -> None:
        assert self.host_tensor is not None and self.device_buffer is not None
        with self._nvtx_range("baseline_numa_unaware"):
            self.device_buffer.copy_(self.host_tensor, non_blocking=False)
            self._synchronize()

    def teardown(self) -> None:
        self.host_tensor = None
        self.device_buffer = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=15, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 1),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if self.host_tensor is None:
            return "Host tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineNUMAUnawareBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=5),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline NUMA latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
