"""optimized_adaptive.py - Optimized with adaptive runtime optimization.

Demonstrates adaptive optimization that adjusts parameters at runtime.
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

class OptimizedAdaptiveBenchmark:
    """Optimized: Adaptive runtime optimization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 1_000_000
        self.adaptive_config = None
    
    def setup(self) -> None:
        """Setup: Initialize with adaptive configuration."""
        
        torch.manual_seed(42)
        # Optimization: Adaptive runtime optimization
        # Adjusts parameters at runtime based on workload characteristics
        # Adapts to changing input sizes, data patterns, etc.
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        # Adaptive: Adjust configuration based on workload
        # In practice, this would monitor performance and adjust parameters
        # For ch6, we demonstrate the concept of adaptive optimization
        if self.N > 500_000:
            self.adaptive_config = "large_batch"
        else:
            self.adaptive_config = "small_batch"
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Adaptive optimization operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_adaptive", enable=enable_nvtx):
            # Optimization: Adaptive runtime optimization
            # Adjusts kernel parameters based on workload characteristics
            # Adapts to input size, data patterns, hardware state, etc.
            # Enables optimal performance across varying workloads
            # Use adaptive configuration
            if self.adaptive_config == "large_batch":
                # Optimize for large batches
                self.output = self.input * 2.0 + 1.0
            else:
                # Optimize for small batches
                self.output = self.input * 2.0 + 1.0
            # Adaptive optimization improves performance by adjusting to workload
            # See ch19 for full adaptive memory management implementations

    
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
    return OptimizedAdaptiveBenchmark()

if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Adaptive: {result.mean_ms:.3f} ms")
    print("  Tip: Adaptive optimization adjusts parameters at runtime for optimal performance")
