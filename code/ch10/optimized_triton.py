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

from common.python import triton_compat  # noqa: F401

import torch
import torch.nn as nn

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

from typing import Optional

from common.python.compile_utils import enable_tf32
from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)
from ch10.workload_config import WORKLOAD, is_smoke_test

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch10")
    return torch.device("cuda")

if TRITON_AVAILABLE:

    @triton.jit
    def fused_residual_relu(
        dst_ptr,
        src_ptr,
        residual_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid = tl.program_id(axis=0)
        offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        values = tl.load(src_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        residual = tl.load(residual_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        fused = tl.maximum(values + residual, 0.0)
        tl.store(dst_ptr + offsets, fused.to(tl.float16), mask=mask)

class OptimizedTritonBenchmark(Benchmark):
    """Optimized: Triton for high-performance custom kernels.
    
    Triton: Uses Triton DSL to write optimized CUDA kernels.
    Provides high-level abstractions for efficient kernel development.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self.residual = None
        self.output = None
        self.use_triton = TRITON_AVAILABLE
        self.workload = WORKLOAD
        self.smoke_test = is_smoke_test()
        self.batch = self.workload.triton_batch_for_mode(self.smoke_test)
        self.micro_batches = self.workload.triton_micro_batches_for_mode(self.smoke_test)
        self.hidden_dim = self.workload.hidden_dim
        self.ffn_dim = self.workload.ffn_dim
        self.total_rows = self.batch * self.micro_batches
        self._checksum = 0.0
    
    def setup(self) -> None:
        """Setup: Initialize model with Triton optimization."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()
        torch.manual_seed(42)
        # Optimization: Triton for custom kernel development
        # Triton provides high-level DSL for writing optimized CUDA kernels
        
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, self.ffn_dim, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.ffn_dim, self.hidden_dim, bias=True),
        ).to(self.device).half().eval()
        
        self.input = torch.randn(
            self.total_rows,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self.residual = torch.randn_like(self.input)
        self.output = torch.empty_like(self.input)

        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with Triton optimization."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("triton", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Triton for custom kernel development
                # Uses Triton DSL to write optimized CUDA kernels
                # Triton: high-level abstractions for efficient kernel development
                
                assert self.model is not None
                assert self.input is not None
                assert self.residual is not None
                assert self.output is not None

                activations = self.model(self.input)
                if self.use_triton and TRITON_AVAILABLE:
                    grid = lambda meta: (triton.cdiv(activations.numel(), meta["BLOCK_SIZE"]),)
                    fused_residual_relu[grid](  # type: ignore[misc]
                        self.output,
                        activations,
                        self.residual,
                        activations.numel(),
                        BLOCK_SIZE=4096,
                        num_warps=4,
                        num_stages=2,
                    )
                else:
                    torch.add(
                        activations,
                        self.residual,
                        alpha=1.0,
                        out=self.output,
                    )
                    torch.relu_(self.output)
                self._checksum = float(self.output.sum().item())
                torch.cuda.synchronize()
                
                # Optimization: Triton benefits
                # - High-level DSL for kernel development
                # - Optimized CUDA kernel generation
                # - Better performance through Triton optimizations
                # - Efficient kernel development workflow

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        self.residual = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
            enable_memory_tracking=False,
            measurement_timeout_seconds=180,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None or self.residual is None or self.output is None:
            return "Buffers missing"
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
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")

if __name__ == "__main__":
    main()
