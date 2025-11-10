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
import torch.nn.functional as F
from common.python import triton_compat  # noqa: F401

try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    tl = None

from typing import Optional, Callable
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

class OptimizedTritonBenchmark(Benchmark):
    """Optimized: Triton when available, otherwise a fused PyTorch kernel."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.input2 = None
        self.output = None
        self.N = 4_000_000
        self.block = 131_072
        self.backend = "pytorch"
        self._compiled_kernel: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    
    def _supports_triton(self) -> bool:
        if not TRITON_AVAILABLE:
            return False
        major, _ = torch.cuda.get_device_capability(self.device)
        return major < 12  # Triton/ptxas does not yet target SM 12.x reliably.
    
    def _pytorch_kernel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        out = x + y
        out = F.silu(out)
        return out * 1.5 + 0.1
    
    def setup(self) -> None:
        """Setup: Initialize tensors and pick backend."""
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.input2 = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self.backend = "triton" if self._supports_triton() else "pytorch"
        if self.backend == "pytorch":
            compiler = getattr(torch, "compile", None)
            if compiler is not None:
                self._compiled_kernel = compiler(self._pytorch_kernel, mode="reduce-overhead")
            else:
                self._compiled_kernel = self._pytorch_kernel
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations using Triton or fused PyTorch kernel."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("triton", enable=enable_nvtx):
            if self.backend == "triton":
                grid = lambda meta: (triton.cdiv(self.N, meta['BLOCK_SIZE']),)
                triton_add_kernel[grid](
                    self.input,
                    self.input2,
                    self.output,
                    self.N,
                    BLOCK_SIZE=1024,
                )
                self.output = self._pytorch_kernel(self.output, torch.zeros_like(self.output))
                torch.cuda.synchronize()
            else:
                assert self._compiled_kernel is not None
                for start in range(0, self.N, self.block):
                    end = min(start + self.block, self.N)
                    lhs = self.input[start:end]
                    rhs = self.input2[start:end]
                    fused = self._compiled_kernel(lhs, rhs)
                    self.output[start:end].copy_(fused)
                torch.cuda.synchronize()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.input2 = None
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
    print(f"\nOptimized Triton: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    if TRITON_AVAILABLE:
        print("  Tip: Triton enables efficient GPU kernel programming with Python-like syntax")
    else:
        print("  WARNING: Triton not available - install with: pip install triton")
