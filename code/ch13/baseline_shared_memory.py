"""baseline_shared_memory.py - Baseline without shared memory optimization in training.

Demonstrates operations without using shared memory for data reuse.
Shared memory: This baseline does not use shared memory optimization.
All data access goes through global memory, causing poor cache utilization.
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

from common.python.compile_utils import compile_model

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class BaselineSharedMemoryBenchmark(Benchmark):
    """Baseline: No shared memory - direct global memory access.
    
    Shared memory: This baseline does not use shared memory optimization.
    All data access goes through global memory, causing poor cache utilization.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize model and data in global memory."""
        torch.manual_seed(42)
        # Baseline: No shared memory optimization
        # Shared memory allows fast data reuse within thread blocks
        # This baseline uses global memory for all data access
        
        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device)
        
        self.model.train()
        
        # Data in global memory (no shared memory optimization)
        self.input = torch.randn(32, 1024, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations without shared memory."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_shared_memory", enable=enable_nvtx):
            # Baseline: No shared memory - all data access via global memory
            # Shared memory would cache frequently accessed data
            # This baseline accesses global memory repeatedly (inefficient)
            
            # Multiple operations accessing same data from global memory
            output1 = self.model(self.input)
            output2 = self.model(self.input)  # Re-access from global memory
            output3 = self.model(self.input)  # Re-access from global memory
            
            # Baseline: No shared memory benefit
            # Global memory access is slower than shared memory
            # Poor cache utilization for repeated data access
            _ = output1 + output2 + output3

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineSharedMemoryBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineSharedMemoryBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: shared_memory")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
