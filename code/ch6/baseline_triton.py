"""baseline_triton.py - Baseline without Triton optimization.

Demonstrates operations using standard PyTorch kernels (not Triton).
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


class BaselineTritonBenchmark(Benchmark):
    """Baseline: Standard PyTorch operations (not Triton kernels)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.bias = None
        self.output = None
        self.N = 4_000_000
        self.chunk = 4096
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.bias = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()

    def _transform(self, window: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        out = window + bias
        out = F.silu(out)
        return out * 1.5 + 0.1
    
    def benchmark_fn(self) -> None:
        """Benchmark: Standard PyTorch operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("triton", enable=enable_nvtx):
            for start in range(0, self.N, self.chunk):
                end = min(start + self.chunk, self.N)
                chunk = self.input[start:end]
                bias = self.bias[start:end]
                out = self._transform(chunk, bias)
                self.output[start:end].copy_(out)
            torch.cuda.synchronize()

    
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
    return BaselineTritonBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Triton (PyTorch): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print("  Note: Uses PyTorch's default kernels, not Triton-optimized")
