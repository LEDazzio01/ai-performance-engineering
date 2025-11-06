"""optimized_moe.py - Optimized Mixture of Experts in AI optimization context.

Demonstrates MoE architecture with sparse activation.
MoE: Uses Mixture of Experts where only subset of experts activate per input.
Sparse activation reduces computation while maintaining model capacity.
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


class MoELayer(nn.Module):
    """Mixture of Experts layer with sparse activation."""
    
    def __init__(self, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router: determines which experts to use
        self.router = nn.Linear(hidden_dim, num_experts)
        
        # Multiple expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with sparse expert activation."""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Flatten for routing
        x_flat = x.view(-1, hidden_dim)
        
        # Router: compute expert weights
        router_logits = self.router(x_flat)  # (batch*seq, num_experts)
        
        # Select top-k experts (sparse activation)
        top_k_weights, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_weights = F.softmax(top_k_weights, dim=-1)
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process with selected experts (sparse activation)
        for i in range(self.num_experts):
            # Find inputs that selected this expert
            expert_mask = (top_k_indices == i).any(dim=-1)
            if expert_mask.any():
                expert_input = x_flat[expert_mask]
                expert_output = self.experts[i](expert_input)
                
                # Weight by router confidence
                expert_weights = top_k_weights[expert_mask]
                expert_idx_in_topk = (top_k_indices[expert_mask] == i).nonzero(as_tuple=True)[1]
                weights = expert_weights.gather(1, expert_idx_in_topk.unsqueeze(1)).squeeze(1)
                
                output[expert_mask] += expert_output * weights.unsqueeze(1)
        
        # Reshape back
        return output.view(batch_size, seq_len, hidden_dim)


class OptimizedMoeBenchmark(Benchmark):
    """Optimized: Mixture of Experts with sparse activation.
    
    MoE: Uses Mixture of Experts architecture where only subset of experts activate per input.
    Sparse activation reduces computation while maintaining model capacity.
    """
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        # Optimization: Compile model for kernel fusion and optimization
        try:
            model = torch.compile(None, mode="reduce-overhead", backend="inductor")
        except Exception:
            pass  # Fallback to eager if compilation fails

        # Optimization: Compile model for kernel fusion and optimization
        try:
            self.model = torch.compile(None, mode="reduce-overhead", backend="inductor")
        except Exception:
            pass  # Fallback to eager if compilation fails

        self.input = None
        self.hidden_dim = 256
    
    def setup(self) -> None:
        """Setup: Initialize MoE model."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: MoE architecture
        # Only subset of experts activate per input (sparse activation)
        # Reduces computation while maintaining model capacity
        
        self.model = nn.Sequential(
            MoELayer(self.hidden_dim, num_experts=4, top_k=2),  # MoE layer
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        
        self.model = compile_model(self.model)
        self.model = self.model.to(self.device)
        # Optimization: Use FP16 for faster computation
        if self.device.type == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                pass  # Fallback to FP32 if FP16 not supported
        self.model.eval()
        
        self.input = torch.randn(4, 32, self.hidden_dim, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: MoE model inference."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_moe", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: MoE with sparse activation
                # Only top-k experts activate per input (e.g., top-2 of 4 experts)
                # Reduces computation while maintaining model capacity
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
    print(f"Optimized: moe")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
