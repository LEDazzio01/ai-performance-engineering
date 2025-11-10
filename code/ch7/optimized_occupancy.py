"""optimized_occupancy.py - Optimized high occupancy in memory access/GEMM context.

Demonstrates operations with high GPU occupancy.
Occupancy: Uses large batch size to maximize GPU occupancy.
High occupancy improves GPU utilization and performance.
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

class OptimizedOccupancyBenchmark(Benchmark):
    """Optimized: High occupancy - maximum GPU utilization.
    
    Occupancy: Uses large batch size to maximize GPU occupancy.
    High occupancy improves GPU utilization and performance.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.requests = None
        self.num_requests = 512
        self.batch_size = 128
        self.feature_dim = 1024
    
    def setup(self) -> None:
        """Setup: Initialize model with large input (high occupancy)."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: High occupancy - large input size
        # Occupancy measures GPU utilization (active warps / max warps)
        # This baseline uses large input causing high occupancy
        
        self.model = nn.Sequential(
            nn.Linear(self.feature_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, self.feature_dim),
        ).to(self.device).eval()
        
        self.requests = torch.randn(self.num_requests, self.feature_dim, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with high occupancy."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_occupancy", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: High occupancy
                # Large input provides sufficient work per kernel
                # Occupancy: high GPU utilization due to large batch size
                for start in range(0, self.num_requests, self.batch_size):
                    batch = self.requests[start:start + self.batch_size]
                    output = self.model(batch)
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
        if self.requests is None:
            return "Requests not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedOccupancyBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedOccupancyBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Occupancy")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
