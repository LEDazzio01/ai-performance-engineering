"""optimized_moe.py - Optimized Mixture of Experts in infrastructure/OS tuning context.

Demonstrates MoE optimized for hardware capabilities.
MoE: Uses Mixture of Experts with hardware-aware expert routing.
Sparse activation optimized for GPU hardware characteristics.
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
import torch.nn.functional as F

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch3")
    return torch.device("cuda")

class MoELayer(nn.Module):
    """Mixture of Experts layer optimized for hardware."""
    
    def __init__(self, hidden_dim: int = 256, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router (gating network) - MoE: expert selection
        self.router = nn.Linear(hidden_dim, num_experts)
        
        # Experts optimized for hardware - MoE: sparse activation
        self.experts = nn.ModuleList([
            nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim * 2),
        nn.ReLU(),
        nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MoE routing optimized for hardware."""
        batch_size, hidden_dim = x.shape
        
        # Router output (MoE: expert selection)
        router_logits = self.router(x)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k expert selection (MoE: sparse activation)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Process with selected experts (MoE: sparse computation optimized for hardware)
        output = torch.zeros_like(x)
        for expert_idx in range(self.num_experts):
            pass
    # Find samples that have this expert in their top_k selection
            expert_mask = (top_k_indices == expert_idx).any(dim=-1)
            if expert_mask.any():
                pass
        expert_input = x[expert_mask]
        expert_output = self.experts[expert_idx](expert_input)
                
        # Weight by router probability (MoE: weighted combination)
        # Get probabilities for this expert from the filtered samples
        expert_probs = top_k_probs[expert_mask]  # Shape: [masked_batch, top_k]
        expert_indices_masked = top_k_indices[expert_mask]  # Shape: [masked_batch, top_k]
        # Find which top_k positions correspond to this expert
        expert_positions = (expert_indices_masked == expert_idx)  # Shape: [masked_batch, top_k]
        # Sum probabilities where this expert was selected
        expert_weight = (expert_probs * expert_positions.float()).sum(dim=-1, keepdim=True)
        output[expert_mask] += expert_output * expert_weight
        
        return output

class OptimizedMoeBenchmark(Benchmark):
    """Optimized: Mixture of Experts optimized for hardware.
    
    MoE: Uses Mixture of Experts with hardware-aware expert routing.
    Sparse activation optimized for GPU hardware characteristics.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None

        self.input = None
    
    def setup(self) -> None:
        """Setup: Initialize MoE model."""
        
        torch.manual_seed(42)
        # Optimization: MoE (Mixture of Experts)
        # MoE uses sparse activation optimized for hardware
        # Only selected experts process each input (efficient)
        
        hidden_dim = 256
        self.model = nn.Sequential(
            MoELayer(hidden_dim, num_experts=4, top_k=2),  # MoE: sparse activation
        ).to(self.device).eval()
        
        self.input = torch.randn(32, hidden_dim, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: MoE model."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_moe", enable=enable_nvtx):
            with torch.no_grad():
                pass
        # Optimization: MoE (Mixture of Experts)
        # MoE: sparse activation - only selected experts process input
        # Optimized for hardware capabilities
        output = self.model(self.input)
                
        # Optimization: MoE benefits
        # - Sparse activation (MoE)
        # - Hardware-aware expert routing
        # - Efficient computation through sparse MoE
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

