"""baseline_coalescing.py - Uncoalesced memory access pattern (baseline).

Demonstrates poor memory access patterns that prevent memory coalescing.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass


from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class BaselineCoalescingBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol.
    
    Demonstrates uncoalesced memory access by accessing elements with stride.
    This prevents memory coalescing and reduces bandwidth utilization.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 50_000_000  # 50M elements - match optimized scale
        self.stride = 32  # Large stride prevents coalescing
    
    def setup(self) -> None:
        """Setup: Initialize tensors (EXCLUDED from timing)."""
        torch.manual_seed(42)
        # Create input tensor
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        # Preallocate output tensor
        self.output = torch.empty((self.N + self.stride - 1) // self.stride, 
                                 device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Uncoalesced memory access pattern.
        
        Accesses memory with large stride, preventing coalescing.
        Each thread accesses non-consecutive memory locations.
        """
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_coalescing_uncoalesced", enable=enable_nvtx):
            # Uncoalesced access: threads access elements with stride
            # This prevents memory coalescing into single 128-byte transactions
            idx = torch.arange(0, self.N, self.stride, device=self.device)
            self.output = self.input[idx] * 2.0

    
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
            return "Output tensor is None"
        if self.input is None:
            return "Input tensor is None"
        # Check that output has correct shape
        expected_shape = (self.N + self.stride - 1) // self.stride
        if self.output.shape[0] != expected_shape:
            return f"Output shape mismatch: expected {expected_shape}, got {self.output.shape[0]}"
        # Check that output values are reasonable (input * 2.0)
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
    print(f"\nBaseline Coalescing Benchmark Results:")
    print(f"  Mean time: {result.mean_ms:.3f} ms")
    print(f"  Std dev: {result.std_ms:.3f} ms")
    print(f"  Min time: {result.min_ms:.3f} ms")
    print(f"  Max time: {result.max_ms:.3f} ms")

