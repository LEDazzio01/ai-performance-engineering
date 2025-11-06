"""baseline_streams.py - Sequential kernel execution (baseline) in infrastructure/OS tuning context.

Demonstrates sequential kernel execution without streams.
Streams: This baseline does not use CUDA streams.
Kernels execute sequentially without overlap.
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
        raise RuntimeError("CUDA required for ch3")
    return torch.device("cuda")


class BaselineStreamsBenchmark(Benchmark):
    """Sequential execution - no overlap.
    
    Streams: This baseline does not use CUDA streams.
    Kernels execute sequentially without overlap.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data1 = None
        self.data2 = None
        self.data3 = None
        self.N = 5_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        # Baseline: Sequential execution (no streams)
        # CUDA streams allow concurrent kernel execution
        # This baseline does not use streams
        
        self.data1 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.data2 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.data3 = torch.randn(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential kernel execution."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_streams", enable=enable_nvtx):
            # Baseline: Sequential execution - kernels run one after another
            # Streams would allow concurrent execution
            # This baseline does not use streams
            self.data1 = self.data1 * 2.0
            torch.cuda.synchronize()  # Wait for completion
            
            self.data2 = self.data2 * 2.0
            torch.cuda.synchronize()  # Wait for completion
            
            self.data3 = self.data3 * 2.0
            torch.cuda.synchronize()  # Wait for completion
            
            # Baseline: No streams benefits
            # Sequential execution (inefficient)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data1 = None
        self.data2 = None
        self.data3 = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=30,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data1 is None:
            return "Data1 tensor not initialized"
        if self.data2 is None:
            return "Data2 tensor not initialized"
        if self.data3 is None:
            return "Data3 tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineStreamsBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=30, warmup=5)
    )
    benchmark = BaselineStreamsBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Streams")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

