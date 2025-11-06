"""optimized_bank_conflicts.py - Optimized bank conflicts with CUDA graphs in kernel launches context.

Demonstrates optimized bank conflict avoidance with CUDA graphs.
Bank conflicts: Avoids bank conflicts through optimized access patterns.
Uses CUDA graphs to reduce kernel launch overhead.
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
        raise RuntimeError("CUDA required for ch12")
    return torch.device("cuda")


class OptimizedBankConflictsBenchmark(Benchmark):
    """Optimized: Avoids bank conflicts with CUDA graphs.
    
    Bank conflicts: Avoids bank conflicts through optimized access patterns.
    Uses CUDA graphs to reduce kernel launch overhead.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.output = None
        self.graph = None
        self.N = 1_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors and CUDA graph."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Avoid bank conflicts with CUDA graphs
        # Bank conflicts: eliminated through contiguous access
        # CUDA graphs: capture kernels to reduce launch overhead
        
        self.data = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        
        # CUDA graphs: warm-up before capture for stable graph creation
        # Warm-up iterations ensure CUDA kernels are compiled and cached
        for _ in range(3):
            output_warmup = self.data * 2.0 + 1.0
            output_warmup = torch.relu(output_warmup)
        torch.cuda.synchronize()
        
        # CUDA graphs: capture optimized operations with static input copy
        # Bank conflicts: contiguous access avoids conflicts
        # Create static copies for graph capture (graph captures tensor addresses)
        self.data_static = self.data.clone()
        self.output_static = torch.empty_like(self.output)
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            # Contiguous access (no bank conflicts)
            self.output_static = self.data_static * 2.0 + 1.0
            self.output_static = torch.relu(self.output_static)
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Bank conflict avoidance with CUDA graphs."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_bank_conflicts", enable=enable_nvtx):
            # Optimization: Bank conflicts avoided with CUDA graphs
            # Bank conflicts: contiguous access eliminates conflicts
            # CUDA graphs: replay captured kernels (low overhead)
            # Copy input to static buffer before replay (graph uses static addresses)
            self.data_static.copy_(self.data)
            self.graph.replay()
            # Copy result back from static buffer
            self.output.copy_(self.output_static)
            
            # Optimization: Bank conflicts and CUDA graphs benefits
            # - No bank conflicts (contiguous access)
            # - Reduced kernel launch overhead (CUDA graphs)
            # - Better performance through graph replay

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        self.output = None
        self.data_static = None
        self.output_static = None
        self.graph = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None or self.output is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedBankConflictsBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedBankConflictsBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Bank Conflicts")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

