"""baseline_bank_conflicts.py - Baseline with bank conflicts in kernel efficiency/arithmetic intensity context.

Demonstrates shared memory bank conflicts.
Bank conflicts: This baseline has bank conflicts in shared memory access.
Multiple threads access the same memory bank simultaneously, causing serialization.
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


class BaselineBankConflictsBenchmark(Benchmark):
    """Baseline: Shared memory bank conflicts.
    
    Bank conflicts: This baseline has bank conflicts in shared memory access.
    Multiple threads access the same memory bank simultaneously, causing serialization.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.output = None
        self.N = 1_000_000
        self.bank_width = 32
        self.conflict_sweeps = 12
        self.tile_elems = 0
        self.conflict_indices = None
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        # Baseline: Bank conflicts in shared memory
        # Bank conflicts occur when multiple threads access the same memory bank
        # This baseline simulates bank conflicts through stride access pattern
        
        self.data = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        self.tile_elems = (self.N // self.bank_width) * self.bank_width
        if self.tile_elems == 0:
            raise RuntimeError("Dataset too small for bank-conflict simulation")
        stride = 17  # coprime with 32 -> forces conflicts
        base_idx = torch.arange(self.bank_width, device=self.device)
        self.conflict_indices = (base_idx * stride % self.bank_width).view(1, -1)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations with bank conflicts."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_bank_conflicts", enable=enable_nvtx):
            # Baseline: deliberately replays the same tile through conflict-heavy gathers.
            active_elems = self.tile_elems
            rows = active_elems // self.bank_width
            data_view = self.data[:active_elems].view(rows, self.bank_width)
            out_view = self.output[:active_elems].view(rows, self.bank_width)
            gather_indices = self.conflict_indices.expand(rows, -1)

            staging = data_view
            for sweep in range(self.conflict_sweeps):
                conflict_tile = torch.gather(staging, 1, gather_indices)
                neighbor = torch.roll(conflict_tile, shifts=1, dims=1)
                staging = conflict_tile * 1.0002 + neighbor * 0.5

            out_view.copy_(staging)
            if active_elems < self.N:
                self.output[active_elems:] = self.data[active_elems:] * 2.0

    
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
