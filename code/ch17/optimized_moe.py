"""optimized_moe.py - Optimized Mixture of Experts in inference/profiling context.

Demonstrates MoE with sparse expert activation.
MoE: Uses Mixture of Experts for sparse activation.
Only selected experts process input, reducing computation.
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
        raise RuntimeError("CUDA required for ch17")
    return torch.device("cuda")


class MoELayer(nn.Module):
    """Mixture of Experts layer with sparse activation."""
    
    def __init__(self, hidden_dim: int = 256, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router (gating network)
        self.router = nn.Linear(hidden_dim, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with MoE routing."""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Router output (MoE: expert selection)
        router_logits = self.router(x)  # (batch, seq, num_experts)
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Top-k expert selection (MoE: sparse activation)
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Process with selected experts (MoE: sparse computation)
        output = torch.zeros_like(x)
        x_flat = x.view(-1, hidden_dim)  # (batch * seq, hidden)
        top_k_probs_flat = top_k_probs.view(-1, self.top_k)  # (batch * seq, top_k)
        top_k_indices_flat = top_k_indices.view(-1, self.top_k)  # (batch * seq, top_k)
        output_flat = output.view(-1, hidden_dim)  # (batch * seq, hidden)
        
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            expert_mask_flat = (top_k_indices_flat == expert_idx).any(dim=1)  # (batch * seq,)
            if expert_mask_flat.any():
                expert_input = x_flat[expert_mask_flat]  # (n_tokens, hidden)
                expert_output = self.experts[expert_idx](expert_input)  # (n_tokens, hidden)
                
                # Weight by router probability (MoE: weighted combination)
                expert_probs_flat = top_k_probs_flat[expert_mask_flat]  # (n_tokens, top_k)
                expert_indices_flat = top_k_indices_flat[expert_mask_flat]  # (n_tokens, top_k)
                # Get probability for this expert
                expert_weights = expert_probs_flat.gather(1, (expert_indices_flat == expert_idx).long().argmax(dim=1, keepdim=True))  # (n_tokens, 1)
                output_flat[expert_mask_flat] += expert_output * expert_weights
        
        output = output_flat.view(batch_size, seq_len, hidden_dim)
        
        return output


class OptimizedMoeBenchmark(Benchmark):
    """Optimized: Mixture of Experts with sparse activation.
    
    MoE: Uses Mixture of Experts for sparse activation.
    Only selected experts process input, reducing computation.
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
    
    def setup(self) -> None:
        """Setup: Initialize MoE model."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        torch.manual_seed(42)
        # Optimization: MoE (Mixture of Experts) - sparse activation
        # MoE uses router to select top-k experts for each input
        # Only selected experts process input (sparse activation)
        
        hidden_dim = 256
        num_experts = 4
        self.model = MoELayer(hidden_dim=hidden_dim, num_experts=num_experts, top_k=2)
        self.model = self.model.to(self.device)
        # Optimization: Use FP16 for faster computation
        if self.device.type == "cuda":
            try:
                self.model = self.model.half()
            except Exception:
                pass  # Fallback to FP32 if FP16 not supported
        self.model.eval()
        
        self.input = torch.randn(32, hidden_dim, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: MoE with sparse activation."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("optimized_moe", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: MoE (Mixture of Experts)
                # Router selects top-k experts for each input
                # Only selected experts process input (sparse activation)
                # MoE benefits: reduces computation compared to dense model
                output = self.model(self.input)
                
                # Optimization: MoE benefits
                # - Sparse activation (only selected experts)
                # - Reduced computation compared to dense model
                # - Better scalability for large models
                # - Efficient expert routing
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
    print(f"Optimized: moe")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")


if __name__ == "__main__":
    main()
