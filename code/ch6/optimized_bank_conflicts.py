"""optimized_bank_conflicts.py - Padding solution to eliminate bank conflicts (optimized).

Demonstrates padding solution to eliminate shared memory bank conflicts.
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
from ch6.cuda_extensions import load_bank_conflicts_extension

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")

class OptimizedBankConflictsBenchmark(Benchmark):
    """Padding solution - no bank conflicts (uses CUDA extension)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 8_000_000
        self._extension = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors and load CUDA extension."""
        
        # Load CUDA extension (will compile on first call)
        self._extension = load_bank_conflicts_extension()
        
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
        # Warm up optimized padded kernel before timing.
        self._extension.bank_conflicts_padded(self.output, self.input)
        torch.cuda.synchronize()
        torch.manual_seed(42)
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Padding eliminates bank conflicts."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("bank_conflicts", enable=enable_nvtx):
            # Call CUDA extension kernel with padding
            self._extension.bank_conflicts_padded(self.output, self.input)
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
        if self.output.shape != self.input.shape:
            return f"Shape mismatch: input={self.input.shape}, output={self.output.shape}"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedBankConflictsBenchmark()

if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nOptimized Bank Conflicts (CUDA Extension): {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
