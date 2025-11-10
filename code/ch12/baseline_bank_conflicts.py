"""baseline_bank_conflicts.py - Baseline with bank conflicts in kernel launches/CUDA graphs context.

Demonstrates bank conflicts with many kernel launches (no CUDA graphs).
Bank conflicts: This baseline has bank conflicts in shared memory access.
Multiple kernel launches without CUDA graphs optimization.
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
from ch6.cuda_extensions import load_bank_conflicts_extension


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch12")
    return torch.device("cuda")


class BaselineBankConflictsBenchmark(Benchmark):
    """Baseline: Bank conflicts with many kernel launches (no CUDA graphs).
    
    Bank conflicts: This baseline has bank conflicts in shared memory access.
    Multiple kernel launches without CUDA graphs optimization.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.output = None
        self._extension = None
        self.N = 1 << 20
        self.launches_per_iter = 64
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        self._extension = load_bank_conflicts_extension()
        self.data = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty_like(self.data)
        # Warm up to include JIT + L2 residency cost in baseline path.
        for _ in range(2):
            self._extension.bank_conflicts(self.output, self.data)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Bank conflicts with many kernel launches."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_bank_conflicts", enable=enable_nvtx):
            # Baseline: Bank conflicts with many kernel launches
            # Bank conflicts: stride access pattern causes conflicts
            # No CUDA graphs: each operation is a separate kernel launch
            if self._extension is None:
                raise RuntimeError("CUDA extension not initialized")
            for _ in range(self.launches_per_iter):
                self._extension.bank_conflicts(self.output, self.data)
            torch.cuda.synchronize(self.device)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        self.output = None
        self._extension = None
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
    return BaselineBankConflictsBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineBankConflictsBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Bank Conflicts")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
