"""optimized_occupancy.py - High occupancy optimization in MoE inference context.

Demonstrates high occupancy for better GPU utilization during MoE inference.
Occupancy: Optimized for high occupancy (many threads per SM).
Maximizes GPU utilization and improves inference performance.
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


class OptimizedOccupancyBenchmark(Benchmark):
    """High occupancy - many threads per SM.
    
    Occupancy: Optimized for high occupancy (many threads per SM).
    Maximizes GPU utilization and improves inference performance.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize model with high occupancy configuration."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: High occupancy configuration
        # Occupancy measures active threads per SM
        # High occupancy maximizes GPU resource utilization
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),  # Larger hidden size = higher occupancy
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # Large batch size maximizes occupancy
        self.input = torch.randn(128, 1024, device=self.device)  # Large batch
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: High occupancy - large work per forward pass."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("occupancy", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Large forward pass - high occupancy
                # Process large amount of work per pass
                # Maximizes threads per SM for better utilization
                
                # Single large forward pass - high occupancy
                # Processes all data at once, maximizing parallelism
                _ = self.model(self.input)
                
                # Optimization: High occupancy benefits
                # - Many threads per SM (high occupancy)
                # - Better GPU resource utilization
                # - Improved inference performance through parallelism
                # - Better hides memory latency with more active threads

    
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
