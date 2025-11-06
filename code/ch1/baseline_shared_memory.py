"""baseline shared memory - Baseline with bank conflicts. Implements Benchmark protocol for harness integration."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch

try:
    import ch1.arch_config  # noqa: F401 - Apply chapter defaults
except ImportError:
    pass

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch1")
    return torch.device("cuda")


class BaselineSharedMemoryBenchmark(Benchmark):
    """Baseline: Shared memory access with bank conflicts.
    
    Shared memory: This baseline has bank conflicts in shared memory access.
    Multiple threads access same bank simultaneously, serializing memory accesses.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 50_000_000  # Match optimized scale
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        # Baseline: Shared memory access with bank conflicts
        # Bank conflicts occur when multiple threads access same bank
        # This baseline simulates bank conflicts through strided access
        
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Shared memory access with bank conflicts."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_shared_memory", enable=enable_nvtx):
            # Baseline: Bank conflicts in shared memory access
            # Strided access pattern causes bank conflicts
            # Multiple threads access same bank simultaneously
            stride = 32  # Stride that causes bank conflicts
            indices = torch.arange(0, self.N, stride, device=self.device)
            self.output = self.input[indices] * 2.0
            
            # Baseline: Bank conflict issues
            # Strided access causes serialization
            # Inefficient shared memory utilization

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
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
        if self.input is None or self.output is None:
            return "Tensors not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineSharedMemoryBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Shared Memory: {result.mean_ms:.3f} ms")
