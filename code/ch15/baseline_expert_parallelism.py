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

from typing import Optional

from common.python.benchmark_harness import (
    Benchmark,
    BenchmarkConfig,
)


def resolve_device() -> torch.device:
    """Return CUDA device if available."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch15")
    return torch.device("cuda")


class ExpertLayer(nn.Module):
    """Single expert in the MoE model."""

    def __init__(self, hidden_size: int = 256, expansion: int = 2):
        super().__init__()
        self.expert = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expansion),
            nn.ReLU(),
            nn.Linear(hidden_size * expansion, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.expert(x)


class DenseMoELayer(nn.Module):
    """Naive MoE that executes *all* experts for every token."""

    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.experts = nn.ModuleList([ExpertLayer(hidden_size) for _ in range(num_experts)])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        outputs = torch.zeros_like(tokens)
        # Every expert processes the full batch which is prohibitively expensive.
        for expert in self.experts:
            outputs += expert(tokens)
        return outputs / len(self.experts)


class BaselineExpertParallelismBenchmark(Benchmark):
    """Baseline: MoE without expert parallelism (all experts on single GPU)."""

    def __init__(self):
        self.device = resolve_device()
        self.model: Optional[nn.Module] = None
        self.input_tokens: Optional[torch.Tensor] = None
        # Large dense workload to highlight the cost of executing every expert.
        self.num_experts = 32
        self.hidden_size = 768
        self.batch_size = 4096

    def setup(self) -> None:
        """Setup: Initialize dense MoE where every expert runs for every token."""
        torch.manual_seed(42)
        dense_layer = DenseMoELayer(self.hidden_size, self.num_experts).to(self.device).eval()
        self.model = dense_layer

        self.input_tokens = torch.randn(
            self.batch_size,
            self.hidden_size,
            device=self.device,
            dtype=torch.float32,
        )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Run every expert sequentially for each token."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.model is None or self.input_tokens is None:
            raise RuntimeError("Model/input not initialized")

        with nvtx_range("baseline_expert_parallelism", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.input_tokens)
        torch.cuda.synchronize()

    def teardown(self) -> None:
        """Cleanup: Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_config(self) -> BenchmarkConfig:
        """Return benchmark configuration."""
        return BenchmarkConfig(
            iterations=6,
            warmup=2,
        )
    
    def validate_result(self) -> Optional[str]:
        """Validate benchmark result."""
        if self.model is None:
            return "Dense MoE not initialized"
        if self.input_tokens is None:
            return "Input tokens not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return BaselineExpertParallelismBenchmark()


def main():
    """Run baseline expert parallelism benchmark."""
    benchmark = get_benchmark()
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
    print(f"Experts: {benchmark.num_experts} (all on single GPU)")
    print("Processing: Sequential expert processing (no expert parallelism)")


if __name__ == "__main__":
    main()
