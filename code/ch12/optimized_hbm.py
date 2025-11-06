"""optimized_hbm.py - Optimized HBM memory access with CUDA graphs in kernel launches context.

Demonstrates HBM optimization with CUDA graphs to reduce kernel launch overhead.
HBM: Optimizes memory access patterns for HBM high bandwidth.
Uses CUDA graphs to capture and replay HBM-optimized operations efficiently.
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
        raise RuntimeError("CUDA required for ch12")
    return torch.device("cuda")

class OptimizedHbmBenchmark(Benchmark):
    """Optimized: HBM memory optimization with CUDA graphs.
    
    HBM: Optimizes memory access patterns for HBM high bandwidth.
    Uses CUDA graphs to capture and replay HBM-optimized operations efficiently.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        # Optimization: Compile model for kernel fusion and optimization
        try:

        self.input = None
        self.input_static = None
        self.output_static = None
        self.graph = None
    
    def setup(self) -> None:
        """Setup: Initialize model with HBM optimization and CUDA graphs."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: HBM memory optimization with CUDA graphs
        # HBM: optimizes for high bandwidth memory
        # CUDA graphs: capture kernels to reduce launch overhead
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        # HBM-optimized memory allocation
        # Large contiguous tensors maximize HBM bandwidth utilization
        self.input = torch.randn(256, 1024, device=self.device)  # Larger batch for HBM
        
        # CUDA graphs: warm-up before capture for stable graph creation
        # Warm-up iterations ensure CUDA kernels are compiled and cached
        for _ in range(3):
            with torch.no_grad():
                output_warmup = self.model(self.input)
                output_warmup = output_warmup.contiguous()
                _ = output_warmup.sum()
        torch.cuda.synchronize()
        
        # CUDA graphs: capture HBM-optimized operations with static buffers
        # HBM: large contiguous operations maximize bandwidth
        # Create static copies for graph capture (graph captures tensor addresses)
        self.input_static = self.input.clone()
        self.output_static = torch.empty_like(self.model(self.input_static))
        
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            with torch.no_grad():
                output = self.model(self.input_static)
                output = output.contiguous()  # Ensure contiguous layout for HBM
                self.output_static.copy_(output)
                _ = self.output_static.sum()
        
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: HBM-optimized operations with CUDA graphs."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_hbm", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: HBM memory optimization with CUDA graphs
                # HBM: large contiguous memory access maximizes bandwidth
                # CUDA graphs: replay captured kernels (low overhead)
                # Copy input to static buffer before replay (graph uses static addresses)
                self.input_static.copy_(self.input)
                self.graph.replay()
                # Result is already in output_static (no copy needed if not used)
                
                # Optimization: HBM and CUDA graphs benefits
                # - High bandwidth memory access (HBM)
                # - Reduced kernel launch overhead (CUDA graphs)
                # - Better performance through graph replay

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.input_static = None
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

