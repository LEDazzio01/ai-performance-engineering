"""baseline ilp basic - Baseline with low instruction-level parallelism. Implements Benchmark protocol for harness integration."""

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


class BaselineIlpBasicBenchmark(Benchmark):
    """Baseline: Sequential operations with low ILP.
    
    ILP: This baseline has low instruction-level parallelism.
    Operations are sequential and dependent, limiting parallel execution.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 100_000_000  # 100M elements - match optimized scale
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        # Baseline: Sequential operations (low ILP)
        # Each operation depends on the previous one
        # Low instruction-level parallelism
        
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential operations with low ILP."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_ilp_basic", enable=enable_nvtx):
            # Baseline: Sequential operations - low ILP
            # Each operation depends on previous one
            # Cannot execute operations in parallel
            val = self.input
            val = val * 2.0      # Op 1
            val = val + 1.0     # Op 2 (depends on Op 1)
            val = val * 3.0     # Op 3 (depends on Op 2)
            val = val - 5.0     # Op 4 (depends on Op 3)
            self.output = val
            
            # Baseline: Low ILP issues
            # Sequential dependencies prevent parallel execution
            # Cannot hide instruction latency

    
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
    return BaselineIlpBasicBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline ILP Basic: {result.mean_ms:.3f} ms")
