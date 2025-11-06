"""baseline_ai_optimization.py - Baseline without AI optimization in storage I/O context.

Demonstrates storage operations without AI-driven optimization.
AI optimization: This baseline does not use AI/ML techniques for optimization.
Uses fixed, heuristic-based approaches.
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
        raise RuntimeError("CUDA required for ch5")
    return torch.device("cuda")


class BaselineAiOptimizationBenchmark(Benchmark):
    """Baseline: Fixed heuristic-based optimization (no AI).
    
    AI optimization: This baseline does not use AI/ML techniques for optimization.
    Uses fixed, heuristic-based approaches.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.data = None
        self.N = 10_000_000
    
    def setup(self) -> None:
        """Setup: Initialize data."""
        torch.manual_seed(42)
        # Baseline: Fixed heuristic-based optimization (no AI)
        # AI optimization uses ML models to predict optimal strategies
        # This baseline uses fixed heuristics
        
        self.data = torch.randn(self.N, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Operations without AI optimization."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_ai_optimization", enable=enable_nvtx):
            # Baseline: Fixed heuristic-based optimization
            # Uses fixed buffer size (no AI prediction)
            # AI optimization would adapt based on learned patterns
            buffer_size = 1024  # Fixed heuristic
            _ = self.data[:buffer_size].sum()
            
            # Baseline: No AI optimization
            # Fixed strategies (not learned/adaptive)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.data = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.data is None:
            return "Data not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineAiOptimizationBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineAiOptimizationBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: Ai Optimization")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
