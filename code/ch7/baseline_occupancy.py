"""baseline_occupancy.py - Baseline with low occupancy in memory access/GEMM context.

Demonstrates operations with low GPU occupancy.
Occupancy: This baseline has low occupancy due to insufficient work per kernel.
Small work per kernel causes GPU underutilization.
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

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch7")
    return torch.device("cuda")


class BaselineOccupancyBenchmark(Benchmark):
    """Baseline: Low occupancy - insufficient work per kernel.
    
    Occupancy: This baseline has low occupancy due to insufficient work per kernel.
    Small work per kernel causes GPU underutilization.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize model with small input (low occupancy)."""
        torch.manual_seed(42)
        # Baseline: Low occupancy - small input size
        # Occupancy measures GPU utilization (active warps / max warps)
        # This baseline uses small input causing low occupancy
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # Small input (low occupancy)
        self.input = torch.randn(2, 1024, device=self.device)  # Very small batch
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with low occupancy."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_occupancy", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Low occupancy
                # Small input causes insufficient work per kernel
                # Occupancy: low GPU utilization due to small batch size
                output = self.model(self.input)
                
                # Baseline: Low occupancy issues
                # Small batch size causes GPU underutilization
                # Not enough work to keep GPU busy
                _ = output.sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
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
    print(f"Baseline: Occupancy")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
