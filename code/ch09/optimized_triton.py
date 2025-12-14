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

from core.benchmark.verification_mixin import VerificationPayloadMixin
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


class OptimizedTritonBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: Triton kernels for efficient GPU operations.
    
    Triton: Uses Triton kernels for optimized GPU operations.
    Provides Python-like syntax for writing efficient CUDA kernels.
    """
    
    def __init__(self):
        super().__init__()
        self.input = None
        self.output = None
        self._output_buffer: Optional[torch.Tensor] = None
        self.N = 1_000_000
        # Triton benchmark - fixed N for kernel comparison
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(self.N),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)

        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self._output_buffer = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self._synchronize()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Triton kernel operations."""
        assert self.input is not None and self._output_buffer is not None
        with self._nvtx_range("triton"):
            if TRITON_AVAILABLE:
                # Optimization: Triton kernel
                # Uses Triton for efficient custom GPU kernels
                # Triton: Python-like syntax for GPU kernel programming
                grid = lambda meta: (triton.cdiv(self.N, meta['BLOCK_SIZE']),)
                triton_kernel[grid](
                    self._output_buffer,
                    self.input,
                    self.N,
                    BLOCK_SIZE=1024,
                )
            else:
                # Fallback: Use optimized PyTorch operations
                self._output_buffer.copy_(self.input * 2.0 + 1.0)
            self.output = self._output_buffer
        self._synchronize()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        self._set_verification_payload(
            inputs={"input": self.input},
            output=self.output.detach().clone(),
            batch_size=self.input.shape[0],
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(1e-5, 1e-5),
        )
            
            # Optimization: Triton benefits
            # - Python-like syntax for GPU kernels
            # - Automatic optimization
            # - Better kernel efficiency
            # - Efficient custom kernel implementation

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        self._output_buffer = None
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


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
