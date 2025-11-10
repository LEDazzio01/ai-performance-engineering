"""Baseline stream-ordered single-GPU (no distributed)."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")


class VectorizationBenchmark(Benchmark):
    """Baseline: Single-GPU stream-ordered (no distributed computing)."""

    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.output = None
        self.weights = None
        self.bias = None
        self.vector_width = 32
        self.num_rows = 32_768

    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        total = self.num_rows * self.vector_width
        self.data = torch.randn(total, device=self.device, dtype=torch.float32).view(self.num_rows, self.vector_width)
        self.output = torch.empty_like(self.data)
        self.weights = torch.randn(self.vector_width, device=self.device, dtype=torch.float32)
        self.bias = torch.randn(self.vector_width, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Scalar-style lane updates with explicit synchronization."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("vectorization_memory", enable=enable_nvtx):
            for lane in range(self.vector_width):
                column = self.data[:, lane]
                transformed = column * self.weights[lane] + self.bias[lane]
                self.output[:, lane] = transformed
                torch.cuda.synchronize()


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        self.output = None
        self.weights = None
        self.bias = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return VectorizationBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Vectorization (Single GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Note: Single-GPU operation, no distributed computing")
