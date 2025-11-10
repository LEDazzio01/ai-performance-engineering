"""baseline_bank_conflicts.py - Shared memory bank conflicts (baseline).

Demonstrates bank conflicts in shared memory access patterns.
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
from common.python.memory_kernels import bank_conflict_transpose


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")


class BaselineBankConflictsBenchmark(Benchmark):
    """Bank conflicts - poor shared memory access pattern.
    
    Bank conflicts occur when multiple threads access the same shared memory bank
    simultaneously, causing serialized access and reduced performance.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 1_048_576  # 1024 * 1024 for square tile transpose
        self.passes = 64
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        self.height = 1024
        self.width = self.N // self.height
        self.input = torch.randn(self.height, self.width, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.width, self.height, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Access pattern causing bank conflicts."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("bank_conflicts", enable=enable_nvtx):
            for _ in range(self.passes):
                bank_conflict_transpose(self.input, self.output, padded=False)
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
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        if self.input is None:
            return "Input tensor not initialized"
        expected = (self.width, self.height)
        if tuple(self.output.shape) != expected:
            return f"Output shape mismatch: expected {expected}, got {tuple(self.output.shape)}"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineBankConflictsBenchmark()


if __name__ == '__main__':
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Bank Conflicts: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
