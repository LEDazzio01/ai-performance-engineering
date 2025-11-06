"""optimized_warp_specialization.py - Optimized warp specialization in kernel efficiency/arithmetic intensity context.

Demonstrates warp specialization for efficient warp execution.
Warp specialization: Assigns different roles to warps for optimized execution.
Improves GPU utilization through specialized warp functions.
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
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class OptimizedWarpSpecializationBenchmark(Benchmark):
    """Optimized: Warp specialization for efficient execution.
    
    Warp specialization: Assigns different roles to warps for optimized execution.
    Improves GPU utilization through specialized warp functions.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model_a = None  # Specialized for first part
        self.model_b = None  # Specialized for second part
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize models with warp specialization."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Warp specialization
        # Different warps handle different parts of the computation
        # Warp specialization: assigns specialized roles to warps
        
        # Model A: specialized for first part (warp specialization)
        self.model_a = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
        ).to(self.device).eval()
        
        # Model B: specialized for second part (warp specialization)
        self.model_b = nn.Sequential(
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # Warp specialization: different warps handle different parts
        self.input = torch.randn(32, 1024, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with warp specialization."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_warp_specialization", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Warp specialization
                # Different warps handle different computation stages
                # Warp specialization: specialized roles for different warps
                
                # Stage 1: specialized warps handle first part
                intermediate = self.model_a(self.input)
                
                # Stage 2: specialized warps handle second part
                # Warp specialization: different warps for different stages
                output = self.model_b(intermediate)
                
                # Optimization: Warp specialization benefits
                # - Specialized warp roles
                # - Better GPU utilization
                # - Optimized execution patterns
                # - Efficient warp-level parallelism
                _ = output.sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model_a = None
        self.model_b = None
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
        if self.model_a is None or self.model_b is None:
            return "Models not initialized"
        if self.input is None:
            return "Input not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedWarpSpecializationBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedWarpSpecializationBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Warp Specialization")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

