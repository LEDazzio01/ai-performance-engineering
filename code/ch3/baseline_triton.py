"""Baseline elementwise kernel using multiple PyTorch launches."""

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
        raise RuntimeError("CUDA required for ch3 triton example")
    return torch.device("cuda")


class BaselineTritonBenchmark(Benchmark):
    """Uses three separate kernels for sin/scale/bias."""

    def __init__(self):
        self.device = resolve_device()
        self.input: Optional[torch.Tensor] = None
        self.scale: Optional[torch.Tensor] = None
        self.bias: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(55)
        self.input = torch.randn(8_388_608, device=self.device, dtype=torch.float16)
        self.scale = torch.full((1,), 1.7, device=self.device, dtype=torch.float16)
        self.bias = torch.full((1,), -0.5, device=self.device, dtype=torch.float16)

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.input is not None and self.scale is not None and self.bias is not None
        with nvtx_range("baseline_triton", enable=enable_nvtx):
            tmp = torch.sin(self.input)
            tmp = tmp * self.scale
            _ = tmp + self.bias

    def teardown(self) -> None:
        self.input = None
        self.scale = None
        self.bias = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=100, warmup=10)

    def validate_result(self) -> Optional[str]:
        if self.input is None:
            return "Input tensor missing"
        return None


def get_benchmark() -> Benchmark:
    return BaselineTritonBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=10, warmup=2),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline Triton latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
