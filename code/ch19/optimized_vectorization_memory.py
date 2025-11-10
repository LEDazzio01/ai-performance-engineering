"""optimized_vectorization_memory.py - Optimized memory management with vectorization.

Demonstrates memory operations optimized with vectorization (SIMD).
Vectorization processes multiple elements per instruction for better efficiency.
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

class OptimizedVectorizationMemoryBenchmark(Benchmark):
    """Optimized: Memory operations with vectorization."""
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.output = None
        self.weights = None
        self.bias = None
        self.vector_width = 32
        self.num_rows = 32_768
    
    def setup(self) -> None:
        """Setup: Initialize tensors for vectorized operations."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Vectorized operations
        # Vectorization uses SIMD instructions to process multiple elements per instruction
        # Improves memory bandwidth utilization and reduces instruction overhead
        # PyTorch operations automatically use vectorization when beneficial
        
        total = self.num_rows * self.vector_width
        self.data = torch.randn(total, device=self.device, dtype=torch.float16).view(self.num_rows, self.vector_width)
        self.output = torch.empty_like(self.data)
        self.weights = torch.randn(1, self.vector_width, device=self.device, dtype=torch.float16)
        self.bias = torch.randn(1, self.vector_width, device=self.device, dtype=torch.float16)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Vectorized memory operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("vectorization_memory", enable=enable_nvtx):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                fused = self.data * self.weights + self.bias
                fused = torch.nn.functional.silu(fused)
                self.output.copy_(fused)
            torch.cuda.synchronize()
            
    # Vectorized operations process multiple elements per instruction
    # Improves memory efficiency compared to scalar operations
    # See ch5 for full vectorization optimization techniques

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        self.output = None
        self.weights = None
        self.bias = None
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
    return OptimizedVectorizationMemoryBenchmark()

if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Vectorization Memory: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(" Tip: Vectorization processes multiple elements per instruction for better memory efficiency")
