"""baseline_low_ai.py - Low arithmetic intensity baseline (baseline).

Memory-bound kernel with low arithmetic intensity.
Many memory operations relative to compute operations.

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


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch13")
    return torch.device("cuda")


class BaselineArithmeticIntensityBenchmark(Benchmark):
    """Low arithmetic intensity baseline - memory-bound."""
    
    def __init__(self):
        self.device = resolve_device()
        self.A = None
        self.B = None
        self.C = None
        self.size = 10_000_000  # Large size to show memory-bound behavior
    
    def setup(self) -> None:
        """Setup: Initialize large tensors."""
        torch.manual_seed(42)
        
        # Large tensors for memory-bound operation
        self.A = torch.randn(self.size, device=self.device, dtype=torch.float32)
        self.B = torch.randn(self.size, device=self.device, dtype=torch.float32)
        self.C = torch.empty_like(self.A)
        
        # Warmup
        self.C = self.A + self.B
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Function to benchmark - low arithmetic intensity (memory-bound)."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_arithmetic_intensity", enable=enable_nvtx):
            # Low arithmetic intensity: simple element-wise operations
            # Many memory operations, few compute operations
            self.C = self.A + self.B  # Memory-bound: read A, read B, write C
            self.C = self.C * 1.5  # Another memory operation
            self.C = self.C - 0.5  # Another memory operation
            # Only 3 arithmetic operations for 3 memory reads + 3 writes
            # Arithmetic Intensity = 3 ops / (3*4 bytes read + 3*4 bytes write) = 3/24 = 0.125

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.A, self.B, self.C
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
            enable_memory_tracking=False,
            enable_profiling=False,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.A is None:
            return "A not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineArithmeticIntensityBenchmark()


if __name__ == "__main__":
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Arithmetic Intensity: {result.mean_ms:.3f} ms")

