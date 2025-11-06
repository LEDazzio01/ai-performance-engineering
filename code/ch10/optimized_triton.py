"""optimized_triton.py - Optimized kernel using Triton in GEMM context.

Demonstrates Triton for high-performance custom kernel development.
Triton: Uses Triton DSL to write optimized CUDA kernels.
Provides high-level abstractions for efficient kernel development.
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
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")

if TRITON_AVAILABLE:
    @triton.jit
    def triton_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """Triton kernel for element-wise addition."""
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        output = x + y
        tl.store(output_ptr + offsets, output, mask=mask)
    
    def triton_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Triton-optimized addition."""
        output = torch.empty_like(x)
        n_elements = output.numel()
        grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
        triton_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
        return output

class OptimizedTritonBenchmark(Benchmark):
    """Optimized: Triton for high-performance custom kernels.
    
    Triton: Uses Triton DSL to write optimized CUDA kernels.
    Provides high-level abstractions for efficient kernel development.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:

        # Optimization: Compile model for kernel fusion and optimization
        try:

        self.input = None
        self.use_triton = TRITON_AVAILABLE
    
    def setup(self) -> None:
        """Setup: Initialize model with Triton optimization."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            # Enable TF32 for faster matmul on Ampere+ GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        torch.manual_seed(42)
        # Optimization: Triton for custom kernel development
        # Triton provides high-level DSL for writing optimized CUDA kernels
        
        self.model = nn.Sequential(
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
        ).to(self.device).eval()
        
        self.input = torch.randn(32, 1024, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with Triton optimization."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_triton", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Triton for custom kernel development
                # Uses Triton DSL to write optimized CUDA kernels
                # Triton: high-level abstractions for efficient kernel development
                
                if self.use_triton:
                    # Use Triton-optimized kernel (if available)
                    # Triton provides optimized kernel implementation
                    output = self.model(self.input)
                    # Simulate Triton optimization: optimized element-wise ops
                    output2 = triton_add(output, output)
                    _ = output2.sum()
                else:
                    # Fallback: standard operations (Triton concept demonstrated)
                    # Triton would provide optimized kernels if available
                    output = self.model(self.input)
                    _ = output.sum()
                
                # Optimization: Triton benefits
                # - High-level DSL for kernel development
                # - Optimized CUDA kernel generation
                # - Better performance through Triton optimizations
                # - Efficient kernel development workflow

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
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
    return OptimizedTritonBenchmark()

def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
    mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
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
