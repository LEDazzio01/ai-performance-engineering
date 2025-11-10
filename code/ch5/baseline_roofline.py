"""Baseline roofline example: memory-bound streaming sum."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch5")
    return torch.device("cuda")


class BaselineRooflineBenchmark(Benchmark):
    """Reads the tensor once with no compute, emphasizing memory bandwidth."""

    def __init__(self):
        self.device = resolve_device()
        self.data: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(2)
        self.data = torch.randn(64_000_000, device=self.device, dtype=torch.float32)

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.data is not None
        with nvtx_range("baseline_roofline", enable=enable_nvtx):
            squared = self.data * self.data
            scaled = squared * 1.5
            activated = torch.sin(scaled)
            _ = activated.mean()

    def teardown(self) -> None:
        self.data = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=40, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.data is None:
            return "Tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineRooflineBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline roofline latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
