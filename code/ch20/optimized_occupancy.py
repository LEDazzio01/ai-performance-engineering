"""optimized_occupancy.py - High occupancy optimization in AI optimization context.

Demonstrates high occupancy for better GPU utilization.
Occupancy: Optimized for high occupancy (many threads per SM).
Maximizes GPU utilization and improves performance.
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


class OptimizedOccupancyBenchmark(Benchmark):
    """High occupancy - many threads per SM.
    
    Occupancy: Optimized for high occupancy (many threads per SM).
    Maximizes GPU utilization and improves performance.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.N = 1_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: High occupancy configuration
        # Occupancy measures active threads per SM
        # High occupancy maximizes GPU resource utilization
        self.data = torch.randn(self.N, dtype=torch.float32, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: High occupancy - large work per kernel."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("occupancy", enable=enable_nvtx):
            # Optimization: Large kernel launches - high occupancy
            # Process large amount of work per kernel
            # Maximizes threads per SM for better utilization
            
            # Single large kernel launch - high occupancy
            # Processes all data at once, maximizing parallelism
            _ = self.data * 2.0
            
            # Optimization: High occupancy benefits
            # - Many threads per SM (high occupancy)
            # - Better GPU resource utilization
            # - Improved performance through parallelism
            # - Better hides memory latency with more active threads

    
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
    print(f"Optimized: occupancy")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
