"""baseline_coalescing.py - Baseline with uncoalesced memory access in occupancy/warp divergence context.

Demonstrates uncoalesced memory access patterns.
Coalescing: This baseline has uncoalesced memory access.
Threads access memory in a non-contiguous pattern, wasting bandwidth.
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
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch8")
    return torch.device("cuda")


class BaselineCoalescingBenchmark(Benchmark):
    """Baseline: Uncoalesced memory access.
    
    Coalescing: This baseline has uncoalesced memory access.
    Threads access memory in a non-contiguous pattern, wasting bandwidth.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 1_000_000
        self.stride = 32  # Large stride prevents coalescing
    
    def setup(self) -> None:
        """Setup: Initialize tensors with uncoalesced access pattern."""
        torch.manual_seed(42)
        # Baseline: Uncoalesced memory access
        # Coalescing requires threads to access contiguous memory
        # This baseline uses large stride, preventing coalescing
        
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Uncoalesced memory access."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_coalescing", enable=enable_nvtx):
            # Baseline: Uncoalesced memory access
            # Access pattern with large stride prevents coalescing
            # Coalescing: threads should access contiguous memory for efficiency
            # This pattern wastes memory bandwidth due to uncoalesced access
            indices = torch.arange(0, self.N, self.stride, device=self.device)
            self.output[indices] = self.input[indices] * 2.0
            
            # Baseline: Uncoalesced access issues
            # Large stride prevents memory coalescing
            # Inefficient memory bandwidth utilization

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input is None or self.output is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineCoalescingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineCoalescingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Coalescing")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

