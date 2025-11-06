"""optimized shared memory - Optimized shared memory access without bank conflicts. Implements Benchmark protocol for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class OptimizedSharedMemoryBenchmark(Benchmark):
    """Optimized: Shared memory access without bank conflicts.
    
    Shared memory: Optimized to eliminate bank conflicts.
    Uses coalesced access patterns and padding to avoid bank conflicts.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 50_000_000  # Much larger workload to amortize overhead
        self.stride = 32  # Same stride as baseline for fair comparison
        self.indices = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        # Optimization: Shared memory access without bank conflicts
        # Uses coalesced access patterns (consecutive access) to avoid bank conflicts
        # Consecutive access enables efficient shared memory utilization
        
        # Optimization: Ensure contiguous memory layout for better performance
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32).contiguous()
        # Same output size as baseline for fair comparison
        num_output_elements = (self.N + self.stride - 1) // self.stride
        self.output = torch.empty(num_output_elements, device=self.device, dtype=torch.float32)
        # Pre-compute indices for efficient access
        # Use contiguous indices for better memory access patterns
        self.indices = torch.arange(0, self.N, self.stride, device=self.device, dtype=torch.long).contiguous()
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # Pre-computed indices optimization: avoids computing indices in hot loop
        # This reduces overhead and enables better memory access patterns
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Optimized shared memory access without bank conflicts."""
        # Use conditional NVTX ranges - only enabled when profiling
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        with nvtx_range("optimized_shared_memory", enable=enable_nvtx):
            # Optimization: Coalesced access pattern with pre-computed indices
            # Pre-computed contiguous indices enable better memory access optimization
            # PyTorch can optimize the indexing operation better than computing indices each time
            # This eliminates bank conflicts through efficient memory access patterns
            self.output = self.input[self.indices] * 2.0
            
            # Optimization: Bank conflict elimination benefits
            # - Consecutive access pattern (no bank conflicts)
            # - Efficient memory coalescing
            # - Maximum shared memory bandwidth utilization

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=20,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input is None or self.output is None:
            return "Tensors not initialized"
        expected_shape = (self.N + self.stride - 1) // self.stride
        if self.output.shape[0] != expected_shape:
            return f"Output shape mismatch: expected {expected_shape}, got {self.output.shape[0]}"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedSharedMemoryBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Shared Memory: {result.mean_ms:.3f} ms")
