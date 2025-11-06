"""Baseline stream-ordered without coalescing optimization."""

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


class BaselineCoalescingStreamsBenchmark(Benchmark):
    """Baseline: Stream-ordered without coalescing optimization."""

    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.stream = None
        self.N = 1_000_000

    def setup(self) -> None:
        """Setup: Initialize stream-ordered allocation."""
        torch.manual_seed(42)
        # Baseline: Stream-ordered without coalescing
        # Memory coalescing optimizes memory access patterns for efficiency
        # This baseline uses stream-ordered allocation but not coalesced access
        self.stream = torch.cuda.Stream()
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Stream-ordered operations without coalescing."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_coalescing_streams", enable=enable_nvtx):
            # Baseline: Stream-ordered without coalescing
            # Memory coalescing ensures threads access contiguous memory
            # This baseline does not optimize memory access patterns
            with torch.cuda.stream(self.stream):
                # Non-coalesced access pattern
                self.output = self.input[::2] * 2.0  # Strided access (not coalesced)
                # Stream-ordered allocation but non-coalesced memory access
                # See ch5 for full coalescing optimizations


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
    return BaselineCoalescingStreamsBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Coalescing Streams: {result.mean_ms:.3f} ms")
    print(" Note: Stream-ordered without memory coalescing optimization")
