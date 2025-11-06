"""baseline_moe_dense.py - Dense MoE routing (baseline).

Uses all experts for every token - inefficient but simple.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to path for imports
repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
    BenchmarkHarness,
    BenchmarkMode
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch16")
    return torch.device("cuda")


class SimpleMoELayer(nn.Module):
    """MoE layer with dense routing (all experts)."""
    
    def __init__(self, hidden_dim, num_experts):
        super().__init__()
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        
        # Experts
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.ReLU(),
                nn.Linear(hidden_dim * 2, hidden_dim)
            )
            for _ in range(num_experts)
        ])
    
    def forward(self, x):
        """Dense routing: Run ALL experts (inefficient)."""
        # Naive: Run all experts and average
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        return torch.stack(expert_outputs).mean(0)


class BaselineMoEBenchmark(Benchmark):
    """Benchmark implementation following Benchmark protocol."""
    
    def __init__(self):
        self.device = resolve_device()
        self.model = None
        self.x = None
        # Reduced workload size to prevent GPU OOM
        self.batch_size = 32  # Reduced to prevent OOM
        self.seq_len = 4096  # Reduced to prevent OOM
        self.hidden_dim = 1024
        self.num_experts = 8
    
    def setup(self) -> None:
        """Setup: initialize model and data."""
        self.model = SimpleMoELayer(hidden_dim=self.hidden_dim, num_experts=self.num_experts)
        self.model = self.model.to(self.device).to(torch.bfloat16).eval()
        self.x = torch.randn(self.batch_size, self.seq_len, self.hidden_dim, device=self.device, dtype=torch.bfloat16)
    
    def benchmark_fn(self) -> None:
        """Function to benchmark."""
        # Use conditional NVTX ranges - only enabled when profiling

        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()

        enable_nvtx = get_nvtx_enabled(config) if config else False


        with nvtx_range("moe_dense", enable=enable_nvtx):
            with torch.no_grad():
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
        """Optional validation."""
        return None


def get_benchmark() -> Benchmark:
    """Factory function for harness discovery."""
    return BaselineMoEBenchmark()


def main() -> None:
    """Standalone execution with timing."""
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5)
    )
    benchmark = BaselineMoEBenchmark()
    result = harness.benchmark(benchmark)
    
    print("=" * 70)
    print("Baseline: Dense MoE Routing (All Experts)")
    print("=" * 70)
    print(f"Model: {benchmark.num_experts} experts, {benchmark.hidden_dim} hidden dim")
    print(f"Batch: {benchmark.batch_size}, SeqLen: {benchmark.seq_len}")
    print("Routing: Dense (all experts for every token)")
    print("Problem: Inefficient - computes unnecessary experts\n")
    
    print(f"Average time: {result.mean_ms:.3f} ms")
    print(f"Median: {result.median_ms:.3f} ms")
    print(f"Std: {result.std_ms:.3f} ms")
    print(f"Experts used: {benchmark.num_experts} (all)")
    print("Status: Dense routing (inefficient)")
    print("\nTip: Use sparse routing (top-k experts) for 2-4x speedup")


if __name__ == "__main__":
    main()
