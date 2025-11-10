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


class TritonBenchmark(Benchmark):
    """Baseline: Single-GPU stream-ordered (no distributed computing)."""

    def __init__(self):
        self.device = resolve_device()
        self.input_a = None
        self.input_b = None
        self.output = None
        self.scatter_idx = None
        self.N = 1_000_000
        self.chunk = 4096

    def setup(self) -> None:
        """Setup: Initialize tensors with random gather pattern."""
        torch.manual_seed(42)
        self.input_a = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.input_b = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self.scatter_idx = torch.randperm(self.N, device=self.device)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Non-coalesced scatter that issues many kernels."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("triton_memory", enable=enable_nvtx):
            for start in range(0, self.N, self.chunk):
                positions = self.scatter_idx[start:start + self.chunk]
                gathered_a = torch.index_select(self.input_a, 0, positions)
                gathered_b = torch.index_select(self.input_b, 0, positions)
                fused = gathered_a * gathered_b + torch.sin(gathered_a)
                self.output.index_copy_(0, positions, fused)
            torch.cuda.synchronize()


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input_a = None
        self.input_b = None
        self.output = None
        self.scatter_idx = None
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
    return TritonBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Triton (Single GPU): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Note: Single-GPU operation, no distributed computing")
