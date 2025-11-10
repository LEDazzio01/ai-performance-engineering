"""HBM baseline: launch dozens of tiny kernels that thrash memory bandwidth."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional, List

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch3")
    return torch.device("cuda")


class BaselineHbmBenchmark(Benchmark):
    """Serializes many small slices, forcing redundant reads from HBM."""

    def __init__(self):
        self.device = resolve_device()
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device)
        self.chunks: List[torch.Tensor] = []

    def setup(self) -> None:
        torch.manual_seed(22)
        big_tensor = torch.randn(4096, 1024, device=self.device, dtype=torch.float32)
        self.chunks = list(big_tensor.chunk(32, dim=0))
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        accumulator = torch.zeros_like(self.chunks[0])
        with nvtx_range("baseline_hbm", enable=enable_nvtx):
            with torch.no_grad():
                for chunk in self.chunks:
                    accumulator += self.model(chunk)
        return accumulator

    def teardown(self) -> None:
        self.chunks = []
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=25, warmup=5)

    def validate_result(self) -> Optional[str]:
        if not self.chunks:
            return "Chunks not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineHbmBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline HBM latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
