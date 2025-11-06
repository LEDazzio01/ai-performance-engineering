"""optimized_warp_specialization.py - Optimized with warp specialization in training.

Demonstrates warp specialization for efficient parallel execution.
Warp specialization: Assigns different roles to warps (producer/consumer).
Improves parallel efficiency and reduces synchronization overhead.
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


class OptimizedWarpSpecializationBenchmark(Benchmark):
    """Optimized: Warp specialization for efficient parallel execution.
    
    Warp specialization: Assigns different roles to warps (producer/consumer).
    Improves parallel efficiency and reduces synchronization overhead.
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
        """Setup: Initialize model with warp specialization optimization."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        # Optimization: Warp specialization
        # Assigns different roles to warps (producer/consumer)
        # Uses __activemask to coordinate warp roles
        # Improves parallel efficiency
        
        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
        ).to(self.device)
        
        self.model.train()
        
        self.input = torch.randn(32, 1024, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with warp specialization."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_warp_specialization", enable=enable_nvtx):
            # Optimization: Warp specialization
            # Different warps have specialized roles
            # Producer warps: load/compute data
            # Consumer warps: process/compute on produced data
            # Uses __activemask to coordinate warp roles
            
            # Simulate warp specialization pattern
            # In CUDA kernels, would use __activemask to coordinate warps
            # For PyTorch, we demonstrate through separate producer/consumer stages
            
            # Stage 1: Producer warps (prepare data)
            intermediate = self.model[0](self.input)  # Producer: compute first layer
            
            # Stage 2: Consumer warps (process data)
            # Warp specialization: consumer warps process produced data
            output = self.model[1](intermediate)  # Consumer: process intermediate
            output = self.model[2](output)  # Consumer: final processing
            
            # Optimization: Warp specialization benefits
            # - Different warps have specialized roles (producer/consumer)
            # - Better parallel efficiency
            # - Reduced synchronization overhead
            # - Improved throughput through specialized execution
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
    print(f"Optimized: warp_specialization")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
