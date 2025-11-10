"""NUMA-unaware baseline: copies pageable CPU tensors to GPU each step."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


class BaselineNUMAUnawareBenchmark(Benchmark):
    """Allocates pageable host memory and blocks on every copy."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.device.type != "cuda":
            raise RuntimeError("CUDA required for NUMA benchmark")
        self.host_tensor: Optional[torch.Tensor] = None
        self.device_buffer: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(9)
        self.host_tensor = torch.randn(128_000_000, dtype=torch.float32)  # ~512 MB
        self.device_buffer = torch.empty_like(self.host_tensor, device=self.device)

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.host_tensor is not None and self.device_buffer is not None
        with nvtx_range("baseline_numa_unaware", enable=enable_nvtx):
            self.device_buffer.copy_(self.host_tensor, non_blocking=False)
            torch.cuda.synchronize()

    def teardown(self) -> None:
        self.host_tensor = None
        self.device_buffer = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=15, warmup=3)

    def validate_result(self) -> Optional[str]:
        if self.host_tensor is None:
            return "Host tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineNUMAUnawareBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline NUMA latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
