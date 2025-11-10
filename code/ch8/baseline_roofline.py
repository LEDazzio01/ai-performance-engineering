"""baseline_roofline.py - Baseline without roofline analysis in occupancy/warp divergence context.

Demonstrates operations without roofline analysis for performance optimization.
Roofline: This baseline does not use roofline analysis.
Does not measure or optimize based on compute/memory bottlenecks.
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
        raise RuntimeError("CUDA required for ch8")
    return torch.device("cuda")


class BaselineRooflineBenchmark(Benchmark):
    """Baseline: Operations without roofline analysis.
    
    Roofline: This baseline does not use roofline analysis.
    Does not measure or optimize based on compute/memory bottlenecks.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.inputs = None
        self.hidden_dim = 1024
        self.batch_size = 256
        self.micro_batches = 4
    
    def setup(self) -> None:
        """Setup: Initialize model without roofline analysis."""
        torch.manual_seed(42)
        # Baseline: No roofline analysis
        # Roofline analysis identifies compute-bound vs memory-bound operations
        # This baseline does not perform roofline analysis
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
        ).to(self.device).eval()
        
        self.inputs = [
            torch.randn(self.batch_size, self.hidden_dim, device=self.device, dtype=torch.float32)
            for _ in range(self.micro_batches)
        ]
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations without roofline analysis."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_roofline", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: No roofline analysis
                # Does not measure arithmetic intensity or identify bottlenecks
                # No optimization based on compute/memory characteristics
                for micro_input in self.inputs:
                    output = self.model(micro_input)
                    _ = output.sum()
            torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.inputs = None
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
        if not self.inputs:
            return "Inputs not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineRooflineBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineRooflineBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Roofline")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
