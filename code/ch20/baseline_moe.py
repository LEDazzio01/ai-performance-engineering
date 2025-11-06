"""baseline_moe.py - Baseline without Mixture of Experts in AI optimization context.

Demonstrates dense model (all parameters active) vs MoE (sparse activation).
MoE: This baseline does not use Mixture of Experts architecture.
All parameters are active for every input, causing inefficient computation.
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

from common.python.compile_utils import compile_model

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch20")
    return torch.device("cuda")


class BaselineMoeBenchmark(Benchmark):
    """Baseline: Dense model (no MoE - all parameters active).
    
    MoE: This baseline does not use Mixture of Experts architecture.
    All parameters are active for every input, causing inefficient computation.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.input = None
        self.hidden_dim = 256
    
    def setup(self) -> None:
        """Setup: Initialize dense model."""
        torch.manual_seed(42)
        # Baseline: Dense model (no MoE)
        # MoE uses sparse activation - only subset of experts active per input
        # This baseline uses all parameters for every input
        
        # Dense feedforward layer (all parameters active)
        self.model = nn.Sequential(
            nn.Linear(self.hidden_dim, 1024),  # Large dense layer
            nn.ReLU(),
            nn.Linear(1024, self.hidden_dim),
        )
        
        self.model = compile_model(self.model)
        self.model = self.model.to(self.device).eval()
        
        self.input = torch.randn(4, 32, self.hidden_dim, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Dense model inference."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_moe", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Dense model - all parameters active
                # MoE would activate only subset of experts per input
                # This baseline computes with all parameters (inefficient)
                output = self.model(self.input)

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.model = None
        self.input = None
        torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=10,
            warmup=2,
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
