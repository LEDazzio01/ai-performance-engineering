"""baseline_warp_divergence_ilp.py - Baseline ILP with warp divergence.

Demonstrates ILP operations that cause warp divergence, limiting parallelism.
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
        raise RuntimeError("CUDA required for ch6")
    return torch.device("cuda")


class BaselineWarpDivergenceILPBenchmark(Benchmark):
    """Baseline: ILP limited by warp divergence."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 1_000_000
    
    def setup(self) -> None:
        """Setup: Initialize tensors."""
        torch.manual_seed(42)
        # Baseline: ILP operations with warp divergence
        # Warp divergence occurs when threads in a warp take different paths
        # This limits instruction-level parallelism
        self.input = torch.randint(0, 2, (self.N,), device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: ILP operations with warp divergence."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_warp_divergence_ilp", enable=enable_nvtx):
            # Baseline: ILP limited by warp divergence
            # Warp divergence causes serialization of divergent paths
            # This reduces instruction-level parallelism
            # Threads in same warp take different execution paths
            mask = self.input > 0.5
            # Divergent branches reduce ILP as paths are serialized
            result1 = self.input * 2.0
            result2 = self.input * 0.5
            self.output = torch.where(mask, result1, result2)
            # Warp divergence limits ILP by serializing divergent execution paths

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.input = None
        self.output = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=100,
            warmup=10,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.output is None:
            return "Output tensor not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineWarpDivergenceILPBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline Warp Divergence ILP: {result.mean_ms:.3f} ms")
    print("  Note: Warp divergence limits instruction-level parallelism")
