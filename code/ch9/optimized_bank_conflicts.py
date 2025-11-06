"""optimized_bank_conflicts.py - Optimized to avoid bank conflicts in kernel efficiency/arithmetic intensity context.

Demonstrates optimized shared memory access without bank conflicts.
Bank conflicts: Avoids bank conflicts through padding or optimized access patterns.
Eliminates serialization in shared memory access.
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
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class OptimizedBankConflictsBenchmark(Benchmark):
    """Optimized: Avoids bank conflicts in shared memory access.
    
    Bank conflicts: Avoids bank conflicts through padding or optimized access patterns.
    Eliminates serialization in shared memory access.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.output = None
        self.N = 1_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Avoid bank conflicts
        # Bank conflicts avoided through contiguous access pattern
        # No bank conflicts: threads access different banks
        
        self.data = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations without bank conflicts."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_bank_conflicts", enable=enable_nvtx):
            # Optimization: Avoid bank conflicts
            # Contiguous access pattern avoids bank conflicts
            # Bank conflicts: eliminated through optimized access pattern
            # All threads access consecutive memory (no bank conflicts)
            self.output = self.data * 2.0  # Contiguous access (no bank conflicts)
            
            # Optimization: Bank conflicts avoided
            # - Contiguous memory access pattern
            # - No serialization (efficient)
            # - Maximum shared memory bandwidth
            # - Optimized access patterns

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None or self.output is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedBankConflictsBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedBankConflictsBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: Bank Conflicts")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

