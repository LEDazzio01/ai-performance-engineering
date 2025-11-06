"""baseline_moe.py - Baseline dense model (no MoE) in hardware overview context.

Demonstrates dense model where all parameters are active.
MoE: This baseline does not use Mixture of Experts.
All parameters process every input, not optimized for hardware capabilities.
Implements Benchmark protocol for harness integration.
"""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn


from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch2")
    return torch.device("cuda")


class BaselineMoeBenchmark(Benchmark):
    """Baseline: Dense model (no MoE - all parameters active).
    
    MoE: This baseline does not use Mixture of Experts.
    All parameters are active for every input, not optimized for hardware capabilities.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize dense model."""
        torch.manual_seed(42)
        # Baseline: Dense model - all parameters are active for every input
        # MoE (Mixture of Experts) uses sparse activation optimized for hardware
        # This baseline does not use MoE - all parameters are always active
        
        hidden_dim = 256
        self.model = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim),
        ).to(self.device).eval()
        
        self.input = torch.randn(32, hidden_dim, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Dense model (no MoE)."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_moe", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Dense model - all parameters active
                # MoE would use sparse activation optimized for hardware
                # This baseline uses all parameters (inefficient)
                output = self.model(self.input)
                
                # Baseline: No MoE benefits
                # All parameters process every input
                # Not optimized for hardware capabilities
                _ = output.sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=50,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.input is None:
            return "Input not initialized"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineMoeBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = BaselineMoeBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Baseline: moe")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
