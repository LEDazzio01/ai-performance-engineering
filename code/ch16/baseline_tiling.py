"""baseline_tiling.py - Baseline without tiling optimization in MoE context.

Demonstrates matrix operations without tiling optimization.
Tiling: This baseline does not use tiling to optimize memory access patterns.
Processes entire matrices at once, causing poor cache utilization.
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


class BaselineTilingBenchmark(Benchmark):
    """Baseline: No tiling - processes entire matrices.
    
    Tiling: This baseline does not use tiling to optimize memory access patterns.
    Processes entire matrices at once, causing poor cache utilization.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize model without tiling."""
        torch.manual_seed(42)
        # Baseline: No tiling optimization
        # Tiling breaks matrices into smaller tiles for better cache usage
        # This baseline processes entire matrices without tiling
        
        # Large linear layer (no tiling)
        self.model = nn.Linear(2048, 2048).to(self.device).eval()
        
        # Large input (no tiling)
        self.input = torch.randn(64, 2048, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Matrix operations without tiling."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_tiling", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: No tiling - process entire matrix at once
                # Tiling would break computation into smaller tiles
                # This baseline processes full matrix, causing poor cache utilization
                
                # Single large matrix multiplication (no tiling)
                output = self.model(self.input)
                
                # Baseline: No tiling benefits
                # - Processes entire matrix (poor cache utilization)
                # - No reuse of cached data
                # - Inefficient memory access patterns
                _ = output.sum()

    
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
    return BaselineTilingBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineTilingBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: tiling")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
