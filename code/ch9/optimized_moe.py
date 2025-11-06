"""optimized_moe.py - Optimized sparse Mixture of Experts in kernel efficiency/arithmetic intensity context.

Demonstrates sparse MoE routing (only top-k experts active).
MoE: Uses sparse MoE routing where only top-k experts are activated.
More efficient than dense MoE routing.
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


class OptimizedMoeBenchmark(Benchmark):
    """Optimized: Sparse MoE routing (only top-k experts active).
    
    MoE: Uses sparse MoE routing where only top-k experts are activated.
    More efficient than dense MoE routing.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.experts = None
        self.gate = None
        self.input = None
        self.num_experts = 4
        self.top_k = 2  # Sparse MoE: only activate top-2 experts
    
    def setup(self) -> None:
        """Setup: Initialize sparse MoE model."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: Sparse MoE routing
        # MoE (Mixture of Experts) uses multiple expert networks
        # This optimized version activates only top-k experts (sparse routing)
        
        hidden_dim = 256
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            ) for _ in range(self.num_experts)
        ]).to(self.device).eval()
        
        # Gate network for routing (MoE: determines which experts to activate)
        self.gate = nn.Linear(hidden_dim, self.num_experts).to(self.device).eval()
        
        # Sparse MoE: only top-k experts will be activated
        self.input = torch.randn(32, hidden_dim, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sparse MoE computation."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_moe", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: Sparse MoE routing
                # MoE: only activate top-k experts (sparse routing)
                gate_logits = self.gate(self.input)  # (batch, num_experts)
                top_k_values, top_k_indices = torch.topk(gate_logits, self.top_k, dim=-1)
                
                # Sparse MoE: only process top-k experts
                outputs = []
                for i in range(self.top_k):
                    expert_idx = top_k_indices[:, i]  # Which expert for each sample
                    expert_out = torch.zeros_like(self.input)
                    
                    # Process each expert's assigned samples (sparse MoE)
                    for expert_id in range(self.num_experts):
                        mask = (expert_idx == expert_id)
                        if mask.any():
                            expert_out[mask] = self.experts[expert_id](self.input[mask])
                    
                    outputs.append(expert_out)
                
                # Weighted combination of top-k experts (sparse MoE)
                gate_weights = torch.softmax(top_k_values, dim=-1)
                output = sum(w.unsqueeze(-1) * out for w, out in zip(gate_weights.unbind(dim=-1), outputs))
                
                # Optimization: Sparse MoE benefits
                # - Only top-k experts active (efficient)
                # - Reduced computation compared to dense MoE
                # - Better kernel efficiency
                # - Sparse activation pattern
                _ = output.sum()

    
    def teardown(self) -> None:
        """Teardown: Clean up resources."""
        self.experts = None
        self.gate = None
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
    return OptimizedMoeBenchmark()


def main() -> None:
    """Standalone execution (for testing)."""
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=50, warmup=5)
    )
    benchmark = OptimizedMoeBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print(f"Optimized: MoE")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()

