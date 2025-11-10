"""baseline_occupancy.py - Low occupancy baseline in MoE inference context.

Demonstrates low occupancy causing GPU underutilization during MoE inference.
Occupancy: This baseline has low occupancy (too few threads per SM).
Causes GPU underutilization and poor inference performance.
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
        raise RuntimeError("CUDA required for ch16")
    return torch.device("cuda")


class BaselineOccupancyBenchmark(Benchmark):
    """Low occupancy - too few threads per SM.
    
    Occupancy: This baseline has low occupancy (too few threads per SM).
    Causes GPU underutilization and poor inference performance.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize model with low occupancy configuration."""
        torch.manual_seed(42)
        # Baseline: Low occupancy configuration
        # Occupancy measures active threads per SM
        # Low occupancy means GPU resources are underutilized
        
        self.model = nn.Sequential(
            nn.Linear(1024, 256),  # Small hidden size = low occupancy
            nn.ReLU(),
            nn.Linear(256, 1024),
        ).to(self.device).eval()
        
        # Small batch size causes low occupancy
        self.input = torch.randn(2, 1024, device=self.device)  # Small batch
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Low occupancy - small work per forward pass."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("occupancy", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Many small forward passes - low occupancy
                # Each pass processes small amount of work
                # Causes low occupancy: too few threads per SM
                
                for _ in range(10):
                    # Small kernel launch - low occupancy
                    _ = self.model(self.input)
                    torch.cuda.synchronize()  # Synchronize after each small batch
                
                # Baseline: Low occupancy issues
                # - Too few threads per SM
                # - GPU resources underutilized
                # - Poor inference performance due to limited parallelism

    
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
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
