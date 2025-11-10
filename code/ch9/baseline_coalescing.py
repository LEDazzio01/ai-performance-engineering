"""baseline_coalescing.py - Uncoalesced memory access (baseline).

Demonstrates poor memory access patterns that prevent memory coalescing.
Implements Benchmark protocol for harness integration.
"""

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
    BenchmarkHarness,
    BenchmarkMode,
)
from common.python.memory_kernels import uncoalesced_copy


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class BaselineCoalescingBenchmark(Benchmark):
    """Uncoalesced memory access - poor pattern."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 10_000_000
        # Large stride prevents coalescing
        self.stride = 32
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty_like(self.input)
        torch.cuda.synchronize()
        uncoalesced_copy(self.input, self.output, self.stride)
        torch.cuda.synchronize()
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty_like(self.input)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Uncoalesced memory access pattern."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("baseline_coalescing_uncoalesced", enable=enable_nvtx):
            # Launch REAL CUDA kernel with uncoalesced stride pattern
            uncoalesced_copy(self.input, self.output, self.stride)
            torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        if self.input is None:
            return "Input tensor not initialized"
        if self.output.shape[0] != self.N:
            return f"Output shape mismatch: expected {self.N}, got {self.output.shape[0]}"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineCoalescingBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Coalescing (Uncoalesced): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
