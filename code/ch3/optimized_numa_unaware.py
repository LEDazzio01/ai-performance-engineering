"""NUMA-aware optimization: pinned memory + async copies overlapped with compute."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


class OptimizedNUMAAwareBenchmark(Benchmark):
    """Uses pinned host memory and overlaps copies with reduction kernels."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("CUDA required for NUMA benchmark")
        self.host_tensor: Optional[torch.Tensor] = None
        self.device_buffers: list[torch.Tensor] = []
        self.copy_stream = torch.cuda.Stream()
        self.cur_slot = 0
        self.next_slot = 1
        self.iteration = 0

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

    def _start_copy(self, slot: int) -> None:
        assert self.host_tensor is not None
        with torch.cuda.stream(self.copy_stream):
            self.device_buffers[slot].copy_(self.host_tensor, non_blocking=True)

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.host_tensor is not None
        torch.cuda.current_stream().wait_stream(self.copy_stream)
        with nvtx_range("optimized_numa", enable=enable_nvtx):
            _ = torch.sum(self.device_buffers[self.cur_slot])
        self.host_tensor.add_(1e-4)
        self._start_copy(self.cur_slot)
        self.cur_slot, self.next_slot = self.next_slot, self.cur_slot
        self.iteration += 1

    def teardown(self) -> None:
        self.host_tensor = None
        self.device_buffers = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=15, warmup=3)

    def validate_result(self) -> Optional[str]:
        if self.host_tensor is None:
            return "Host tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedNUMAAwareBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nOptimized NUMA latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
