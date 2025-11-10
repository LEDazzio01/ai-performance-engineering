"""baseline_cuda_graphs.py - Baseline separate kernel launches without CUDA graphs.

Demonstrates separate kernel launches without CUDA graphs.
CUDA graphs: This baseline launches kernels separately without graph capture.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional, List

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class BaselineCudaGraphsBenchmark(Benchmark):
    """Baseline: Separate kernel launches without CUDA graphs."""

    def __init__(self):
        self.device = resolve_device()
        self.layers: Optional[nn.ModuleList] = None
        self.inputs: Optional[List[torch.Tensor]] = None
        self.batch_size = 16
        self.hidden_dim = 256
        self.unrolled_steps = 6

    def setup(self) -> None:
        """Setup: Initialize model."""
        torch.manual_seed(42)
        self.layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim), nn.ReLU()).to(self.device).eval()
                for _ in range(4)
            ]
        )
        self.inputs = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
            for _ in range(8)
        ]
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Separate kernel launches without CUDA graphs."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.layers is None or self.inputs is None:
            raise RuntimeError("CUDA graphs baseline not initialized")

        with nvtx_range("baseline_cuda_graphs", enable=enable_nvtx):
            with torch.no_grad():
                for micro in self.inputs:
                    x = micro
                    for _ in range(self.unrolled_steps):
                        for layer in self.layers:
                            x = layer(x)
        torch.cuda.synchronize()

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.layers = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=12,
            warmup=2,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.layers is None:
            return "Layers not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineCudaGraphsBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
