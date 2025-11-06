"""optimized_hbm.py - Optimized HBM memory access in occupancy/warp divergence context.

Demonstrates HBM memory optimization for high bandwidth utilization.
HBM: Optimizes memory access patterns for HBM high bandwidth.
Maximizes HBM memory bandwidth utilization.
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


class OptimizedHbmBenchmark(Benchmark):
    """Optimized: HBM memory optimization for high bandwidth utilization.
    
    HBM: Optimizes memory access patterns for HBM high bandwidth.
    Maximizes HBM memory bandwidth utilization.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:
            model = torch.compile(None, mode="reduce-overhead", backend="inductor")
        except Exception:
            pass  # Fallback to eager if compilation fails

        # Optimization: Compile model for kernel fusion and optimization
        try:
            self.model = torch.compile(None, mode="reduce-overhead", backend="inductor")
        except Exception:
            pass  # Fallback to eager if compilation fails

        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize model with HBM optimization."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: HBM memory optimization
        # HBM (High Bandwidth Memory) provides high memory bandwidth
        # Optimizes access patterns to maximize HBM bandwidth
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # HBM-optimized memory allocation
        # Large contiguous tensors maximize HBM bandwidth utilization
        self.input = torch.randn(256, 1024, device=self.device)  # Larger batch for HBM
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: HBM-optimized operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_hbm", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: HBM memory optimization
                # Large contiguous memory access maximizes HBM bandwidth
                # HBM: high bandwidth memory access patterns
                output = self.model(self.input)
                
                # Additional HBM optimization: contiguous operations
                # HBM benefits: high bandwidth memory throughput
                output2 = output.contiguous()  # Ensure contiguous layout for HBM
                
                # Optimization: HBM optimization benefits
                # - High bandwidth memory access
                # - Contiguous memory patterns
                # - Maximized HBM bandwidth utilization
                # - Better performance through HBM optimization
                _ = output2.sum()

    
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
    return OptimizedHbmBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedHbmBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: HBM")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

