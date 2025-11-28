"""NUMA-aware optimization: pinned memory + async copies overlapped with compute."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


class OptimizedNUMAAwareBenchmark(BaseBenchmark):
    """Uses pinned host memory and overlaps copies with reduction kernels."""

    def __init__(self):
        super().__init__()
        self.host_tensor: Optional[torch.Tensor] = None
        self.device_buffers: list[torch.Tensor] = []
        self.copy_stream = torch.cuda.Stream()
        self.cur_slot = 0
        self.next_slot = 1
        self.iteration = 0
        bytes_per_iter = 128_000_000 * 2  # float16 bytes
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            bytes_per_iteration=float(bytes_per_iter),
        )

    def setup(self) -> None:
        torch.manual_seed(9)
        self.host_tensor = torch.randn(128_000_000, dtype=torch.float16, pin_memory=True)
        self.device_buffers = [
            torch.empty_like(self.host_tensor, device=self.device),
            torch.empty_like(self.host_tensor, device=self.device),
        ]
        self.cur_slot = 0
        self.next_slot = 1
        self._start_copy(self.cur_slot)
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        self._start_copy(self.next_slot)
        self.iteration = 0
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            bytes_per_iteration=self._workload.bytes_per_iteration,
        )

    def _start_copy(self, slot: int) -> None:
        assert self.host_tensor is not None
        with torch.cuda.stream(self.copy_stream):
            self.device_buffers[slot].copy_(self.host_tensor, non_blocking=True)

    def benchmark_fn(self) -> None:
        assert self.host_tensor is not None
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        with self._nvtx_range("optimized_numa"):
            _ = torch.sum(self.device_buffers[self.cur_slot])
        self.host_tensor.add_(1e-4)
        self._start_copy(self.cur_slot)
        self.cur_slot, self.next_slot = self.next_slot, self.cur_slot
        self.iteration += 1
        self._synchronize()

    def teardown(self) -> None:
        self.host_tensor = None
        self.device_buffers = []
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=15, warmup=5)

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_system_config_metrics
        return compute_system_config_metrics(
            numa_nodes=getattr(self, 'numa_nodes', 1),
            cpu_cores=getattr(self, 'cpu_cores', 64),
        )

    def validate_result(self) -> Optional[str]:
        if self.host_tensor is None:
            return "Host tensor not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedNUMAAwareBenchmark()


if __name__ == "__main__":
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=5),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized NUMA latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
