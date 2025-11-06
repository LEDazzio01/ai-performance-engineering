"""optimized_warp_divergence_ilp.py - Optimized ILP avoiding warp divergence.

Demonstrates ILP optimization by avoiding warp divergence (predication).
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
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")

class OptimizedWarpDivergenceILPBenchmark(Benchmark):
    """Optimized: High ILP by avoiding warp divergence."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 1_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        
        torch.manual_seed(42)
        # Optimization: ILP maximized by avoiding warp divergence
        # Use predication/branchless code to maintain high ILP
        # All threads execute same instructions, avoiding divergence
        self.input = torch.randint(0, 2, (self.N,), device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: High ILP operations avoiding warp divergence."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_warp_divergence_ilp", enable=enable_nvtx):
    # Optimization: High ILP by avoiding warp divergence
    # Use branchless operations that all threads execute
    # Independent operations expose maximum instruction-level parallelism
    # Avoid divergent branches that serialize execution
            
    # Compute both paths independently (high ILP)
            result1 = self.input * 2.0
            result2 = self.input * 0.5
            
    # Combine using predication (no divergence)
            mask = self.input > 0.5
            self.output = torch.where(mask, result1, result2)
    # High ILP: Independent operations executed in parallel
    # No warp divergence: all threads execute same instructions

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedWarpDivergenceILPBenchmark()

if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Warp Divergence ILP: {result.mean_ms:.3f} ms")
    print("  Tip: Avoiding warp divergence maximizes instruction-level parallelism")
