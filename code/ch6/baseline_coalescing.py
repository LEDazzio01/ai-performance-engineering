"""baseline_coalescing.py - Uncoalesced memory access pattern (baseline).

Demonstrates poor memory access patterns that prevent memory coalescing.
Uses PyTorch CUDA extension for accurate GPU timing with CUDA Events.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch


from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode,
)

# Import CUDA extension
from ch6.cuda_extensions import load_coalescing_extension


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")


class BaselineCoalescingBenchmark(Benchmark):
    """Uncoalesced memory access - poor pattern (uses CUDA extension)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 10_000_000
        self.stride = 32  # Large stride prevents coalescing
        self._extension = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""
        # Load CUDA extension (will compile on first call)
        self._extension = load_coalescing_extension()
        
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        # Output size matches stride pattern
        output_size = (self.N + self.stride - 1) // self.stride
        self.output = torch.empty(output_size, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Uncoalesced memory access pattern."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_coalescing_uncoalesced", enable=enable_nvtx):
            # Call CUDA extension kernel with stride
            # The kernel accesses input[access_idx] where access_idx = idx * stride
            # Output size matches number of threads
            num_threads = (self.N + self.stride - 1) // self.stride
            output = torch.empty(num_threads, device=self.device, dtype=torch.float32)
            
            # Call kernel with stride parameter
            self._extension.uncoalesced_copy(output, self.input, self.stride)
            
            self.output = output

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
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
        if self.output is None:
            return "Output tensor not initialized"
        if self.input is None:
            return "Input tensor not initialized"
        expected_shape = (self.N + self.stride - 1) // self.stride
        if self.output.shape[0] != expected_shape:
            return f"Output shape mismatch: expected {expected_shape}, got {self.output.shape[0]}"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineCoalescingBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Coalescing (CUDA Extension): {result.mean_ms:.3f} ms")
