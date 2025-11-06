"""baseline_occupancy.py - Low occupancy baseline in AI optimization context.

Demonstrates low occupancy causing GPU underutilization.
Occupancy: This baseline has low occupancy (too few threads per SM).
Causes GPU underutilization and poor performance.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

from common.python.compile_utils import compile_model

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class BaselineOccupancyBenchmark(Benchmark):
    """Low occupancy - too few threads per SM.
    
    Occupancy: This baseline has low occupancy (too few threads per SM).
    Causes GPU underutilization and poor performance.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.N = 1_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        # Baseline: Low occupancy configuration
        # Occupancy measures active threads per SM
        # Low occupancy means GPU resources are underutilized
        self.data = torch.randn(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Low occupancy - small work per kernel."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_occupancy_low", enable=enable_nvtx):
            # Baseline: Many small kernel launches - low occupancy
            # Each launch processes small amount of work
            # Causes low occupancy: too few threads per SM
            chunk_size = 1000  # Small chunks cause low occupancy
            
            for i in range(0, self.N, chunk_size):
                chunk = self.data[i:i+chunk_size]
                # Small kernel launch - low occupancy
                _ = chunk * 2.0
            
            # Baseline: Low occupancy issues
            # - Too few threads per SM
            # - GPU resources underutilized
            # - Poor performance due to limited parallelism

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineOccupancyBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineOccupancyBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: occupancy")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
