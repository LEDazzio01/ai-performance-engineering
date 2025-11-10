"""baseline_expert_parallelism.py - Baseline MoE without expert parallelism.

Demonstrates Mixture of Experts (MoE) without expert parallelism.
Expert parallelism: This baseline processes all experts sequentially on a single GPU.
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
        raise RuntimeError("CUDA required for ch19")
    return torch.device("cuda")


class ExpertLayer(nn.Module):
    """Single expert in MoE model."""
    
    def __init__(self, hidden_size: int = 256):
        super().__init__()
        self.expert = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert(x)


class BaselineExpertParallelismBenchmark(Benchmark):
    """Baseline: MoE without expert parallelism (all experts on single GPU)."""
    
    def __init__(self):
        self.device = resolve_device()
        self.experts = None
        self.router = None
        self.input_data = None
        self.num_experts = 32
        self.top_k = 4  # Top-k experts per token
        self.batch_tokens = 512
    
    def setup(self) -> None:
        """Setup: Initialize MoE model with all experts on single GPU."""
        torch.manual_seed(42)
        # Baseline: All experts on single GPU
        # Expert parallelism distributes experts across multiple GPUs
        # This baseline processes all experts sequentially on one GPU
        self.experts = nn.ModuleList([
            ExpertLayer(256).to(self.device) for _ in range(self.num_experts)
        ])
        
        # Simple router (gating network) to select top-k experts
        self.router = nn.Linear(256, self.num_experts).to(self.device)
        
        self.input_data = torch.randn(self.batch_tokens, 256, device=self.device)
        torch.cuda.synchronize()
    
    def benchmark_fn(self) -> None:
        """Benchmark: Sequential expert processing without parallelism."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled
        
        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        
        # Baseline: Process all experts sequentially on single GPU
        # No expert parallelism - all experts on one device
        with nvtx_range("baseline_expert_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                router_logits = self.router(self.input_data)  # [batch, num_experts]
                top_k_weights, top_k_indices = torch.topk(router_logits, self.top_k, dim=-1)
                top_k_weights = torch.softmax(top_k_weights, dim=-1)
                
                # Process each expert sequentially
                output = torch.zeros_like(self.input_data)
                for expert_id in range(self.num_experts):
                    expert_mask = (top_k_indices == expert_id).any(dim=-1)
                    if expert_mask.any():
                        expert_input = self.input_data[expert_mask]
                        # Naive dispatch copies activations to host between experts
                        host_copy = expert_input.to("cpu", non_blocking=False)
                        host_copy = F.relu(host_copy * 1.0001)
                        torch.cuda.synchronize()
                        staged = host_copy.to(self.device, non_blocking=False)
                        expert_output = self.experts[expert_id](staged)
                        output[expert_mask] += expert_output
                        torch.cuda.synchronize()
        
        torch.cuda.synchronize()
    
    def teardown(self) -> None:
        """Cleanup: Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=20,
            warmup=3,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.experts is None or len(self.experts) == 0:
            return "Experts not initialized"
        if self.router is None:
            return "Router not initialized"
        if self.input_data is None:
            return "Input data not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineExpertParallelismBenchmark()


if __name__ == '__main__':
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    benchmark = get_benchmark()
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
