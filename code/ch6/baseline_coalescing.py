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
        self.rows = 4096
        self.cols = 2048
        self.N = self.rows * self.cols
        self._extension = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""
        # Load CUDA extension (will compile on first call)
        self._extension = load_coalescing_extension()
        
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        # Output matches full transpose result
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
        # Warm up extension so compilation and first-use costs are outside measurement.
        temp_output = torch.empty_like(self.output)
        self._extension.uncoalesced_copy(temp_output, self.input, self.rows, self.cols)
        torch.cuda.synchronize()
        # Recreate tensors so benchmark iterations start from identical data.
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Uncoalesced memory access pattern."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_coalescing_uncoalesced", enable=enable_nvtx):
            # Naive transpose reads/writes with uncoalesced access
            self._extension.uncoalesced_copy(self.output, self.input, self.rows, self.cols)
            # Synchronize to catch any CUDA errors immediately
            torch.cuda.synchronize()
            
    
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
            setup_timeout_seconds=120,  # CUDA extension compilation can take time
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        if self.input is None:
            return "Input tensor not initialized"
        if self.output.shape[0] != self.N:
            return f"Output shape mismatch: expected {self.N}, got {self.output.shape[0]}"
        output_matrix = self.output.view(self.cols, self.rows)
        reference = self.input.view(self.rows, self.cols).transpose(0, 1).contiguous()
        max_err = (output_matrix - reference).abs().max().item()
        if max_err > 1e-4:
            return f"Transpose mismatch: max error {max_err}"
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
    print(f"\nBaseline Coalescing (CUDA Extension): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
