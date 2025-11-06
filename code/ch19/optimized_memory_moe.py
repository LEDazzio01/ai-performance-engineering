"""optimized_memory_moe.py - Optimized memory management with Mixture of Experts.

Demonstrates MoE architecture with adaptive memory management.
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
    BenchmarkHarness,
    BenchmarkMode,
)

def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")


class Expert(nn.Module):
    """Single expert in MoE architecture."""
    
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.fc2 = nn.Linear(hidden_dim * 4, hidden_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(x)))


class MoELayer(nn.Module):
    """Mixture of Experts layer with adaptive routing."""
    
    def __init__(self, hidden_dim: int, num_experts: int = 4, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Router
        self.router = nn.Linear(hidden_dim, num_experts)
        
        # Experts
        self.experts = nn.ModuleList([Expert(hidden_dim) for _ in range(num_experts)])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with expert routing."""
        batch_size, seq_len, hidden_dim = x.shape
        
        # Compute router logits
        router_logits = self.router(x)  # (batch, seq, num_experts)
        
        # Top-k routing
        top_k_logits, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
        top_k_probs = F.softmax(top_k_logits, dim=-1)
        
        # Flatten for processing
        x_flat = x.view(-1, hidden_dim)  # (batch * seq, hidden)
        router_flat = top_k_probs.view(-1, self.top_k)  # (batch * seq, top_k)
        indices_flat = top_k_indices.view(-1, self.top_k)  # (batch * seq, top_k)
        
        # Process through selected experts
        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            # Find tokens routed to this expert
            mask = (indices_flat == expert_idx).any(dim=1)
            if mask.any():
                expert_output = self.experts[expert_idx](x_flat[mask])
                # Weight by router probability
                weights = router_flat[mask].gather(
                    1, (indices_flat[mask] == expert_idx).long().argmax(dim=1, keepdim=True)
                )
                output[mask] += expert_output * weights
        
        return output.view(batch_size, seq_len, hidden_dim)


class OptimizedMemoryMoEBenchmark(Benchmark):
    """Optimized: MoE with adaptive memory management."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None

        self.x = None
        self.batch_size = 4
        self.seq_len = 1024
        self.hidden_dim = 256
        self.num_experts = 4
    
    def setup(self) -> None:
        """Setup: Initialize MoE model."""
        
        # Optimization: Enable cuDNN benchmarking for optimal kernel selection
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        # MoE layer with adaptive routing
        # Optimization: Use BF16 for faster computation + Tensor Core acceleration
        # BF16 provides 2x memory reduction vs FP32 with better numerical stability than FP16
        self.model = MoELayer(
            hidden_dim=self.hidden_dim,
            num_experts=self.num_experts,
            top_k=2
        )
        self.model = self.model.to(self.device).to(dtype=torch.bfloat16).eval()
        
        self.x = torch.randn(
            self.batch_size, self.seq_len, self.hidden_dim,
            device=self.device, dtype=torch.bfloat16
        )
    
    def benchmark_fn(self) -> None:
        """Benchmark: MoE with adaptive memory management."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False

        with nvtx_range("optimized_memory_moe", enable=enable_nvtx):
            with torch.no_grad():
                # Optimization: MoE allows adaptive memory usage
                # Only selected experts are activated per token
                # This reduces memory footprint compared to dense models
                # BF16 precision provides 2x memory reduction + Tensor Core acceleration
                _ = self.model(self.x)

    
    def teardown(self) -> None:
        """Cleanup."""
        del self.model, self.x
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark-specific config."""
        return BenchmarkConfig(
        iterations=20,
            warmup=5,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Model not initialized"
        if self.x is None:
            return "Input tensor not initialized"
        try:
            with torch.no_grad():
                output = self.model(self.x)
            if output.shape != self.x.shape:
                return f"Output shape mismatch: expected {self.x.shape}, got {output.shape}"
        except Exception as e:
            return f"MoE forward pass failed: {e}"
        return None

def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return OptimizedMemoryMoEBenchmark()

def main() -> None:
    """Standalone execution."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = OptimizedMemoryMoEBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Optimized: Memory MoE (Mixture of Experts)")
    print("=" * 70)
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Num experts: {benchmark.num_experts}, Top-k: 2")
    print(" Tip: MoE enables adaptive memory usage by activating only selected experts")

if __name__ == "__main__":
    main()
