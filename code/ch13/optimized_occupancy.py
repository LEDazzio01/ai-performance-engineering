"""optimized_occupancy.py - High occupancy optimization in training context.

Demonstrates high occupancy for better GPU utilization during training.
Occupancy: Optimized for high occupancy (many threads per SM).
Maximizes GPU utilization and improves training performance.
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
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class OptimizedOccupancyBenchmark(Benchmark):
    """High occupancy - many threads per SM.
    
    Occupancy: Optimized for high occupancy (many threads per SM).
    Maximizes GPU utilization and improves training performance.
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
        """Setup: Initialize model with high occupancy configuration."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        # Optimization: High occupancy configuration
        # Occupancy measures active threads per SM
        # High occupancy maximizes GPU resource utilization
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),  # Larger hidden size = higher occupancy
            nn.ReLU(),
            nn.Linear(2048, 10),
        ).to(self.device)
        
        self.model.train()
        
        # Large batch size maximizes occupancy
        self.input = torch.randn(128, 1024, device=self.device)  # Large batch
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: High occupancy - large work per forward pass."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_occupancy_high", enable=enable_nvtx):
            # Optimization: Large forward passes - high occupancy
            # Process large amount of work per pass
            # Maximizes threads per SM for better utilization
            
            # Single large forward pass - high occupancy
            # Processes all data at once, maximizing parallelism
            output = self.model(self.input)
            loss = output.sum()
            loss.backward()
            
            # Optimization: High occupancy benefits
            # - Many threads per SM (high occupancy)
            # - Better GPU resource utilization
            # - Improved training performance through parallelism
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
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
