"""baseline_ai_optimization.py - Baseline without AI-driven optimization.

Demonstrates operations without AI-driven optimization (manual tuning).
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


class BaselineAIOptimizationBenchmark(Benchmark):
    """Baseline: Manual optimization (no AI-driven optimization)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.input = None
        self.output = None
        self.N = 1_000_000
    
    def setup(self) -> None:
        """Setup: Initialize with manual configuration."""
        torch.manual_seed(42)
        # Baseline: Manual optimization
        # AI-driven optimization uses reinforcement learning/AlphaTensor to find optimal kernels
        # This baseline uses manually tuned parameters
        self.input = torch.randn(self.N, device=self.device, dtype=torch.float32)
        self.output = torch.empty(self.N, device=self.device, dtype=torch.float32)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Manual optimization operations."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_ai_optimization", enable=enable_nvtx):
            # Baseline: Manual optimization
            # AI-driven optimization uses RL/AlphaTensor to discover optimal algorithms
            # This baseline uses manually designed kernels
            # AI optimization can discover better algorithms than manual design
            self.output = self.input * 2.0 + 1.0
            # Manual optimization requires expert knowledge and trial-and-error

    
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
    return BaselineAIOptimizationBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(f"\nBaseline AI Optimization (Manual): {result.mean_ms:.3f} ms")
    print("  Note: Uses manual optimization, not AI-driven algorithm discovery")
