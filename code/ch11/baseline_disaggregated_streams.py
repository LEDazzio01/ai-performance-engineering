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
        raise RuntimeError("CUDA required for ch11")
    return torch.device("cuda")


class DisaggregatedStreamsBenchmark(Benchmark):
    """Baseline: Single-GPU stream-ordered (no distributed computing)."""

    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.stream = None
        self.N = 1_000_000

    def setup(self) -> None:
        """Setup: Initialize single-GPU tensors."""
        torch.manual_seed(42)
        # Baseline: Single-GPU stream-ordered operation
        # Distributed computing uses multiple GPUs for parallel stream-ordered operations
        # This baseline uses only one GPU
        self.stream = torch.cuda.Stream()
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Single-GPU stream-ordered operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_disaggregated_streams", enable=enable_nvtx):
            # Baseline: Single-GPU stream-ordered
            # Distributed computing enables stream-ordered operations across multiple GPUs
            # This baseline processes on single GPU only
            # Distributed stream-ordered enables larger workloads through multi-GPU parallelism
            with torch.cuda.stream(self.stream):
                self.output = self.input * 2.0 + 1.0
                # Single-GPU: Limited by single device's capacity
                # See ch17 for full distributed training implementations


    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
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
    return DisaggregatedStreamsBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Disaggregated streams (Single GPU): {result.mean_ms:.3f} ms")
    print(" Note: Single-GPU operation, no distributed computing")
