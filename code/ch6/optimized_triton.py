"""optimized_triton.py - Optimized with Triton kernels.

Demonstrates operations using Triton for efficient GPU kernel programming.
Triton provides a Python-like language for writing optimized CUDA kernels.
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
    # Fallback if Triton not available
    tl = None

from typing import Optional
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")

if TRITON_AVAILABLE:
    @triton.jit
    def triton_add_kernel(
        x_ptr,
        y_ptr,
        output_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
        ):
        """Triton kernel for element-wise addition."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)

class OptimizedTritonBenchmark:
    """Optimized: Uses Triton kernels for efficient GPU operations."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 1_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        
        torch.manual_seed(42)
        # Optimization: Prepare for Triton kernel execution
        # Triton is a GPU programming language for writing efficient kernels
        # Provides Python-like syntax for CUDA kernel development
        # Enables explicit optimization of memory access and compute patterns
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations using Triton kernels."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_triton", enable=enable_nvtx):
            # Optimization: Use Triton kernel for efficient GPU operations
            # Triton provides Python-like syntax for CUDA kernel development
            # Enables explicit optimization of memory access patterns
            # Blocksize optimization for better GPU utilization
            if TRITON_AVAILABLE:
                grid = lambda meta: (triton.cdiv(self.N, meta['BLOCK_SIZE']),)
                triton_add_kernel[grid](
                    self.input,
                    self.output,
                    self.N,
                    BLOCK_SIZE=1024,
                )
                # Apply additional operation
                self.output = self.output * 2.0 + 1.0
                # Triton kernels enable fine-grained optimization of GPU operations
            else:
                # Fallback if Triton not available
                # Use optimized PyTorch operations
                self.output = self.input * 2.0 + 1.0

    
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
        if self.output is None:
            return "Output tensor not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedTritonBenchmark()

if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Triton: {result.mean_ms:.3f} ms")
    if TRITON_AVAILABLE:
        print("  Tip: Triton enables efficient GPU kernel programming with Python-like syntax")
    else:
        print("  WARNING: Triton not available - install with: pip install triton")
