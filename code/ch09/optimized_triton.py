"""optimized_triton.py - Optimized Triton kernel in kernel efficiency/arithmetic intensity context.

Demonstrates Triton for efficient custom GPU kernels.
Triton: Uses Triton kernels for optimized GPU operations.
Provides Python-like syntax for writing efficient CUDA kernels.
Implements BaseBenchmark for harness integration.
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

from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


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


class OptimizedTritonBenchmark(BaseBenchmark):
    """Optimized: Triton kernels for efficient GPU operations.
    
    Triton: Uses Triton kernels for optimized GPU operations.
    Provides Python-like syntax for writing efficient CUDA kernels.
    """
    
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None
        self.N = 1_000_000
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
    
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
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Triton kernel operations."""
        assert self.input is not None and self.output is not None
        with self._nvtx_range("triton"):
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
        self._synchronize()
            
            # Optimization: Triton benefits
            # - Python-like syntax for GPU kernels
            # - Automatic optimization
            # - Better kernel efficiency
            # - Efficient custom kernel implementation

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        super().teardown()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from core.benchmark.metrics import compute_roofline_metrics
        return compute_roofline_metrics(
            total_flops=float(getattr(self, 'total_flops', getattr(self, 'N', 1024) * 2)),
            total_bytes=float(getattr(self, 'N', 1024) * 4 * 2),
            elapsed_ms=getattr(self, '_last_elapsed_ms', 1.0),
            precision="fp16",
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input is None or self.output is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for harness discovery."""
    return OptimizedTritonBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=100, warmup=10)
    )
    benchmark = OptimizedTritonBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Triton")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
