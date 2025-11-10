"""optimized_shared_memory.py - Optimized with shared memory in MoE context.

Demonstrates shared memory optimization for data reuse.
Shared memory: Uses shared memory to cache frequently accessed data.
Improves cache utilization and reduces global memory access.
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
        raise RuntimeError("CUDA required for ch16")
    return torch.device("cuda")


class OptimizedSharedMemoryBenchmark(Benchmark):
    """Optimized: Shared memory for data reuse."""

    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 1 << 20
        self._extension = None

    def setup(self) -> None:
        """Setup: Initialize model and data with shared memory optimization."""
        torch.manual_seed(42)
        self._extension = load_bank_conflicts_extension()
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty_like(self.input)
        self._extension.bank_conflicts_padded(self.output, self.input)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Operations with shared memory optimization."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self._extension is None or self.input is None or self.output is None:
            raise RuntimeError("Optimized shared-memory benchmark not initialized")

        with nvtx_range("optimized_shared_memory", enable=enable_nvtx):
            self._extension.bank_conflicts_padded(self.output, self.input)

    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=4,
            setup_timeout_seconds=120,
        )

    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.input is None or self.output is None:
            return "Input/output not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedSharedMemoryBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedSharedMemoryBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: shared_memory")
    print("=" * 70)
    print(f"Average time: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
    print(f"Median: {result.timing.median_ms if result.timing else 0.0:.3f} ms")
    print(f"Std: {result.timing.std_ms if result.timing else 0.0:.3f} ms")


if __name__ == "__main__":
    main()
