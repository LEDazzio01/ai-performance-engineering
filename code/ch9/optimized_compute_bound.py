"""optimized_compute_bound.py - Compute-bound kernel (high arithmetic intensity).

Complex math operations with high arithmetic intensity.
AI > 250 FLOP/Byte (compute-bound, exceeds roofline ridge point).
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import triton
import triton.language as tl
from typing import Optional
from common.python import triton_compat  # noqa: F401
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


@triton.jit
def _compute_bound_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    sin_out = tl.sin(x)
    cos_out = tl.cos(x)
    product = sin_out * cos_out
    squared = product * product
    sqrt_term = tl.sqrt(tl.abs(product))
    combined = squared + sqrt_term
    fused = combined * 0.95 + tl.exp(product * 0.001)
    tl.store(y_ptr + offsets, fused, mask=mask)


class OptimizedComputeBoundBenchmark(Benchmark):
    """Compute-bound kernel - high arithmetic intensity through fusion."""
    
    def __init__(self):
        self.device = resolve_device()
        self.data: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.N = 10_000_000  # Same size as baseline
        self.block_size = 4096
    
    def setup(self) -> None:
        """Setup: Initialize tensors and validate fused kernel."""
        torch.manual_seed(42)
        self.data = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.output = torch.empty_like(self.data)
        torch.cuda.synchronize()
        self._validate_kernel_correctness()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Triton-fused operations (single kernel)."""
        if self.data is None or self.output is None:
            raise RuntimeError("CUDA tensors not initialized")

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_compute_bound", enable=enable_nvtx):
            self._launch_kernel(self.data, self.output)
            torch.cuda.synchronize()
            self.data, self.output = self.output, self.data

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        self.output = None
        torch.cuda.empty_cache()
    
    def _launch_kernel(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        grid = lambda META: (triton.cdiv(self.N, META["BLOCK"]),)
        _compute_bound_kernel[grid](
            src,
            dst,
            self.N,
            BLOCK=self.block_size,
        )

    def _validate_kernel_correctness(self) -> None:
        assert self.data is not None
        assert self.output is not None
        reference_input = self.data.clone()
        self._launch_kernel(reference_input, self.output)
        torch.cuda.synchronize()
        reference = self._reference_op(reference_input)
        max_error = torch.max(torch.abs(self.output - reference)).item()
        if max_error > 5e-4:
            raise RuntimeError(f"Optimized compute bound kernel mismatch (max error={max_error:.5f})")

    @staticmethod
    def _reference_op(tensor: torch.Tensor) -> torch.Tensor:
        sin_out = torch.sin(tensor)
        cos_out = torch.cos(tensor)
        product = sin_out * cos_out
        squared = product * product
        sqrt_term = torch.sqrt(torch.abs(product))
        combined = squared + sqrt_term
        return combined * 0.95 + torch.exp(product * 0.001)
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}, got {self.data.shape[0]}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedComputeBoundBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Compute Bound: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
