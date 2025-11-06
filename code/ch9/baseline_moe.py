"""baseline_moe.py - Baseline dense Mixture of Experts in kernel efficiency/arithmetic intensity context.

Demonstrates dense MoE routing (all experts active).
MoE: This baseline uses dense MoE routing where all experts are active.
Inefficient compared to sparse MoE routing.
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
        raise RuntimeError("CUDA required for ch9")
    return torch.device("cuda")


class BaselineMoeBenchmark(Benchmark):
    """Baseline: Dense MoE routing (all experts active).
    
    MoE: This baseline uses dense MoE routing where all experts are active.
    Inefficient compared to sparse MoE routing.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.experts = None
        self.input = None
        self.num_experts = 4
    
    def setup(self) -> None:
        """Setup: Initialize dense MoE model."""
        torch.manual_seed(42)
        # Baseline: Dense MoE routing
        # MoE (Mixture of Experts) uses multiple expert networks
        # This baseline activates all experts (dense routing)
        
        hidden_dim = 256
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ) for _ in range(self.num_experts)
        ]).to(self.device).eval()
        
        # Dense MoE: all experts process all inputs
        self.input = torch.randn(32, hidden_dim, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Dense MoE computation."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("baseline_moe", enable=enable_nvtx):
            with torch.no_grad():
                # Baseline: Dense MoE routing
                # All experts process all inputs (inefficient)
                # MoE: dense routing activates all experts
                outputs = []
                for expert in self.experts:
                    expert_out = expert(self.input)
                    outputs.append(expert_out)
                
                # Average all expert outputs (dense MoE)
                output = torch.stack(outputs).mean(dim=0)
                
                # Baseline: Dense MoE inefficiency
                # All experts active (expensive)
                _ = output.sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.experts = None
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
        if self.experts is None:
            return "Experts not initialized"
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
    print(f"Baseline: MoE")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

