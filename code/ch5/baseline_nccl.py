"""baseline_nccl.py - Baseline without NCCL in storage I/O context.

Demonstrates storage operations without NCCL for multi-GPU communication.
NCCL: This baseline does not use NCCL for collective communication.
Uses CPU-based or inefficient communication patterns.
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
        raise RuntimeError("CUDA required for ch5")
    return torch.device("cuda")


class BaselineNcclBenchmark(Benchmark):
    """Baseline: No NCCL - CPU-based or inefficient communication.
    
    NCCL: This baseline does not use NCCL for collective communication.
    Uses CPU-based or inefficient communication patterns.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.N = 10_000_000
    
    def setup(self) -> None:
        """Setup: Initialize data without NCCL."""
        torch.manual_seed(42)
        # Baseline: No NCCL - CPU-based communication
        # NCCL provides optimized GPU-to-GPU collective communication
        # This baseline does not use NCCL
        
        self.data = torch.randn(self.N, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations without NCCL."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_nccl", enable=enable_nvtx):
            # Baseline: No NCCL - CPU-based communication
            # Simulates inefficient communication without NCCL
            result = self.data.sum()
            
            # Simulate CPU-based communication (no NCCL)
            cpu_result = result.cpu()
            cpu_result = cpu_result * 2.0  # Simulate CPU operation
            final_result = cpu_result.to(self.device)
            
            # Baseline: No NCCL benefits
            # CPU-based communication (inefficient)
            _ = final_result

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineNcclBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineNcclBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Nccl")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
