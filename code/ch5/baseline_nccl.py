"""Chapter 5 NCCL baseline: CPU aggregation for gradients."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.benchmark_harness import Benchmark, BenchmarkConfig


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch5")
    return torch.device("cuda")


class BaselineNcclBenchmark(Benchmark):
    """Copies gradient shards to CPU to mimic NCCL absence."""

    def __init__(self):
        self.device = resolve_device()
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device)
        self.input: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(4)
        self.input = torch.randn(256, 1024, device=self.device)

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.input is not None
        with nvtx_range("baseline_nccl", enable=enable_nvtx):
            out = self.model(self.input)
            shards = torch.chunk(out, 4, dim=0)
            reduced = sum(shard.cpu() for shard in shards)
            _ = reduced.to(self.device)

    def teardown(self) -> None:
        self.input = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=30, warmup=4)

    def validate_result(self) -> Optional[str]:
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> Benchmark:
    return BaselineNcclBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline NCCL latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
