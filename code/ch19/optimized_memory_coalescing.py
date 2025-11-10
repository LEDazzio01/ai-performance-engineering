"""optimized_memory_coalescing.py - Optimized memory management with coalescing.

Demonstrates memory coalescing for efficient memory access patterns.
Coalescing groups memory accesses to maximize memory bandwidth utilization.
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
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")

class OptimizedMemoryCoalescingBenchmark(Benchmark):
    """Optimized: Memory operations with coalescing."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input_matrix = None
        self.output_matrix = None
        self.rows = 2048
        self.cols = 1024
        self.vector_width = 16
        self.scale_vec = None
        self.bias_vec = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors for coalesced operations."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Coalesced memory access
        # Coalescing groups memory accesses to maximize bandwidth utilization
        # PyTorch operations automatically use coalescing when beneficial
        
        self.input_matrix = torch.randn(
            self.rows, self.cols, device=self.device, dtype=torch.float16
        ).contiguous()
        self.output_matrix = torch.empty_like(self.input_matrix)
        self.scale_vec = torch.randn(1, self.vector_width, device=self.device, dtype=torch.float16)
        self.bias_vec = torch.randn(1, self.vector_width, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Coalesced memory operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_memory_coalescing", enable=enable_nvtx):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                view = self.input_matrix.view(-1, self.vector_width)
                fused = view * self.scale_vec + self.bias_vec
                self.output_matrix.view(-1, self.vector_width).copy_(fused)
            torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input_matrix = None
        self.output_matrix = None
        self.scale_vec = None
        self.bias_vec = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output_matrix is None:
            return "Output tensor not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedMemoryCoalescingBenchmark()

if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Memory Coalescing: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Tip: Coalescing groups memory accesses to maximize bandwidth utilization")
