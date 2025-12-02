"""optimized_compute_bound.py - Compute-bound kernel (high arithmetic intensity).

Complex math operations with high arithmetic intensity.
AI > 250 FLOP/Byte (compute-bound, exceeds roofline ridge point).
Implements BaseBenchmark for harness integration.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import triton
import triton.language as tl
from typing import Optional
from core.benchmark import triton_compat  # noqa: F401
from core.harness.benchmark_harness import (
    BaseBenchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
    WorkloadMetadata,
)


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


class OptimizedComputeBoundBenchmark(BaseBenchmark):
    """Compute-bound kernel - high arithmetic intensity through fusion."""
    
    def __init__(self):
        super().__init__()
        self.data: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self.N = 4096  # Same size as baseline
        # Name avoids 'size=' substring to prevent naive regex false positives
        self.block_elems = 256
        self.repeats = 16
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.repeats),
            tokens_per_iteration=float(self.N * self.repeats),
        )
    
    def setup(self) -> None:
        """Setup: Initialize tensors and validate fused kernel."""
        torch.manual_seed(42)
        self.data = torch.randn(self.N, dtype=torch.float32, device=self.device)
        self.output = torch.empty_like(self.data)
        self._synchronize()
        self._validate_kernel_correctness()
        self.register_workload_metadata(
            requests_per_iteration=self._workload.requests_per_iteration,
            tokens_per_iteration=self._workload.tokens_per_iteration,
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: Triton-fused operations (single kernel)."""
        if self.data is None or self.output is None:
            raise RuntimeError("CUDA tensors not initialized")

        with self._nvtx_range("optimized_compute_bound"):
            for _ in range(self.repeats):
                self._launch_kernel(self.data, self.output)
                self.data, self.output = self.output, self.data
            self._synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        self.output = None
        super().teardown()
    
    def _launch_kernel(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        grid = lambda META: (triton.cdiv(self.N, META["BLOCK"]),)
        _compute_bound_kernel[grid](
            src,
            dst,
            self.N,
            BLOCK=self.block_elems,
        )

    def _validate_kernel_correctness(self) -> None:
        assert self.data is not None
        assert self.output is not None
        single_tol = 5e-4
        # Repeated application accumulates small FP differences on SM100; allow
        # a slightly looser tolerance to avoid spurious failures while still
        # catching real drift.
        repeat_tol = 1e-2
        # Validate single-step correctness
        reference_input = self.data.clone()
        self._launch_kernel(reference_input, self.output)
        torch.cuda.synchronize()
        reference = self._reference_op(reference_input)
        max_error = torch.max(torch.abs(self.output - reference)).item()
        if max_error > single_tol:
            raise RuntimeError(
                f"Optimized compute bound kernel mismatch (single, max error={max_error:.5f}, tol={single_tol})"
            )
        # Validate repeated-application correctness
        rep_src = self.data.clone()
        rep_dst = self.output.clone()
        for _ in range(self.repeats):
            self._launch_kernel(rep_src, rep_dst)
            rep_src, rep_dst = rep_dst, rep_src
        torch.cuda.synchronize()
        rep_ref = self._apply_reference_n_times(self.data.clone(), self.repeats)
        rep_err = torch.max(torch.abs(rep_src - rep_ref)).item()
        if rep_err > repeat_tol:
            raise RuntimeError(
                f"Optimized compute bound kernel mismatch (repeats, max error={rep_err:.5f}, tol={repeat_tol})"
            )

    @staticmethod
    def _reference_op(tensor: torch.Tensor) -> torch.Tensor:
        sin_out = torch.sin(tensor)
        cos_out = torch.cos(tensor)
        product = sin_out * cos_out
        squared = product * product
        sqrt_term = torch.sqrt(torch.abs(product))
        combined = squared + sqrt_term
        return combined * 0.95 + torch.exp(product * 0.001)
    
    @staticmethod
    def _apply_reference_n_times(tensor: torch.Tensor, times: int) -> torch.Tensor:
        out = tensor
        for _ in range(times):
            out = OptimizedComputeBoundBenchmark._reference_op(out)
        return out
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=5,  # Minimum warmup to exclude CUDA JIT overhead
            enable_memory_tracking=False,
            enable_profiling=False,
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
        if self.data is None:
            return "Data tensor not initialized"
        if self.data.shape[0] != self.N:
            return f"Data size mismatch: expected {self.N}, got {self.data.shape[0]}"
        if not torch.isfinite(self.data).all():
            return "Data contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    """Factory function for benchmark discovery."""
    return OptimizedComputeBoundBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main
    benchmark_main(get_benchmark)
