"""optimized_distributed.py - Fused virtual data-parallel step on a single GPU."""

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
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch12")
    return torch.device("cuda")


class OptimizedDistributedBenchmark(Benchmark):
    """Optimized: fuse virtual ranks into one CUDA graph replay."""

    def __init__(self):
        self.device = resolve_device()
        self.model: nn.Module | None = None
        self.graph = None
        self.static_input = None
        self.graph_output = None
        self.virtual_ranks = 4
        self.micro_batch = 32
        self.hidden_dim = 1024
        self._last = 0.0

    def setup(self) -> None:
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, 2 * self.hidden_dim),
            nn.ReLU(),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
        ).to(self.device).eval()

        total = self.virtual_ranks * self.micro_batch
        sample = torch.randn(total, self.hidden_dim, device=self.device)
        for _ in range(3):
            with torch.no_grad():
                _ = self.model(sample)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        self.static_input = sample.clone()
        self.graph_output = torch.empty_like(sample)
        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                out = self.model(self.static_input)
                self.graph_output.copy_(out)

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_distributed", enable=enable_nvtx):
            if self.graph is None or self.static_input is None or self.graph_output is None:
                raise RuntimeError("Graph not initialized")
            # Pretend each chunk belongs to a different rank; fuse them before replay.
            fused = torch.randn(
                self.virtual_ranks * self.micro_batch,
                self.hidden_dim,
                device=self.device,
            )
            self.static_input.copy_(fused)
            self.graph.replay()
            self._last = float(self.graph_output.sum())
            torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.model = None
        self.graph = None
        self.static_input = None
        self.graph_output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=50, warmup=5)

    def validate_result(self) -> Optional[str]:
        if self.graph is None:
            return "Graph not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedDistributedBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Distributed: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
