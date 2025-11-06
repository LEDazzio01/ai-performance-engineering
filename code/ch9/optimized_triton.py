"""optimized_triton.py - Optimized Triton kernel in kernel efficiency/arithmetic intensity context.

Demonstrates Triton for efficient custom GPU kernels.
Triton: Uses Triton kernels for optimized GPU operations.
Provides Python-like syntax for writing efficient CUDA kernels.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


if TRITON_AVAILABLE:
    @triton.jit
    def triton_kernel(
        output_ptr,
        input_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Triton kernel for element-wise operation."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        
        # Load input
        input_data = tl.load(input_ptr + offsets, mask=mask)
        
        # Compute: output = input * 2.0 + 1.0
        output_data = input_data * 2.0 + 1.0
        
        # Store output
        tl.store(output_ptr + offsets, output_data, mask=mask)


class OptimizedTritonBenchmark(Benchmark):
    """Optimized: Triton kernels for efficient GPU operations.
    
    Triton: Uses Triton kernels for optimized GPU operations.
    Provides Python-like syntax for writing efficient CUDA kernels.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 1_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Triton kernels
        # Triton provides a Python-like language for writing GPU kernels
        
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Triton kernel operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_triton", enable=enable_nvtx):
            if TRITON_AVAILABLE:
                # Optimization: Triton kernel
                # Uses Triton for efficient custom GPU kernels
                # Triton: Python-like syntax for GPU kernel programming
                grid = lambda meta: (triton.cdiv(self.N, meta['BLOCK_SIZE']),)
                triton_kernel[grid](
                    self.output,
                    self.input,
                    self.N,
                    BLOCK_SIZE=1024,
                )
            else:
                # Fallback: Use optimized PyTorch operations
                # Triton would provide more efficient kernels if available
                # Simulate Triton benefit with optimized PyTorch
                self.output = self.input * 2.0 + 1.0
            
            # Optimization: Triton benefits
            # - Python-like syntax for GPU kernels
            # - Automatic optimization
            # - Better kernel efficiency
            # - Efficient custom kernel implementation

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input is None or self.output is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedTritonBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
    mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=100, warmup=10)
    )
    benchmark = OptimizedTritonBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Triton")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

