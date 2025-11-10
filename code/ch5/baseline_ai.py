"""Baseline AI optimization example: repeated CPU-bound orchestration."""

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
        raise RuntimeError("CUDA required for ch5 ai example")
    return torch.device("cuda")


class TinyBlock(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(x)))


class BaselineAIBenchmark(Benchmark):
    """Runs several tiny blocks sequentially with CPU sync between them."""

    def __init__(self):
        self.device = resolve_device()
        self.blocks = nn.ModuleList(TinyBlock(1024).to(self.device) for _ in range(4))
        self.inputs: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(0)
        self.inputs = torch.randn(512, 1024, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.inputs is not None
        with nvtx_range("baseline_ai", enable=enable_nvtx):
            out = self.inputs
            for block in self.blocks:
                out = block(out)
                torch.cuda.synchronize()
        return out

    def teardown(self) -> None:
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=40, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.inputs is None:
            return "Inputs missing"
        return None


def get_benchmark() -> Benchmark:
    return BaselineAIBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=5, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nBaseline AI latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
