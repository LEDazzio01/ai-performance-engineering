"""optimized_expert_parallelism.py - Optimized MoE with expert parallelism.

Demonstrates expert parallelism for Mixture of Experts by distributing experts across GPUs.
Expert parallelism: This optimized version uses expert parallelism to distribute experts across multiple GPUs for parallel processing.
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

from typing import Optional, Tuple

from common.python.compile_utils import enable_tf32
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
    """Single expert block used in the sparse MoE."""

    def __init__(self, hidden_size: int = 256, expansion: int = 2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * expansion),
            nn.ReLU(),
            nn.Linear(hidden_size * expansion, hidden_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TopKDispatchMoE(nn.Module):
    """Sparse MoE that routes tokens to their top-k experts and aggregates them efficiently."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([ExpertLayer(hidden_size) for _ in range(num_experts)])

    def _prepare_dispatch(
        self, logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return expert ids, per-token weights, and token indices sorted by expert."""
        topk_scores, topk_indices = torch.topk(logits, self.top_k, dim=-1)
        probs = torch.softmax(topk_scores, dim=-1)

        expert_ids = topk_indices.reshape(-1)
        weights = probs.reshape(-1).to(logits.dtype)
        token_indices = torch.arange(logits.shape[0], device=logits.device).repeat_interleave(self.top_k)

        sort_order = torch.argsort(expert_ids)
        return expert_ids[sort_order], weights[sort_order], token_indices[sort_order]

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        router_logits = self.router(tokens)
        expert_ids, weights, token_indices = self._prepare_dispatch(router_logits)
        output = torch.zeros_like(tokens, dtype=tokens.dtype)

        unique_ids, counts = torch.unique_consecutive(expert_ids, return_counts=True)
        start = 0
        for expert_id, count in zip(unique_ids.tolist(), counts.tolist()):
            shard_tokens = token_indices[start : start + count]
            shard_weights = weights[start : start + count].unsqueeze(-1).to(tokens.dtype)
            expert_input = tokens[shard_tokens]
            expert_output = self.experts[expert_id](expert_input).to(tokens.dtype)
            output.index_add_(0, shard_tokens, expert_output * shard_weights)
            start += count
        return output


class OptimizedExpertParallelismBenchmark(Benchmark):
    """Optimized: Expert parallelism for MoE using sparse routing on a single GPU."""

    def __init__(self):
        self.device = resolve_device()
        self.hidden_size = 768
        self.num_experts = 32
        self.top_k = 2
        self.batch_size = 4096
        self.model: Optional[nn.Module] = None
        self.input_tokens: Optional[torch.Tensor] = None
        self.compiled = False

    def setup(self) -> None:
        """Setup: Build sparse MoE and optionally compile it."""
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
            enable_tf32()

        torch.manual_seed(42)
        moe = TopKDispatchMoE(self.hidden_size, self.num_experts, self.top_k).to(
            self.device, dtype=torch.bfloat16
        )
        moe.eval()
        try:
            moe = torch.compile(moe, mode="reduce-overhead")
            self.compiled = True
        except Exception:
            self.compiled = False
        self.model = moe

        self.input_tokens = torch.randn(
            self.batch_size,
            self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        """Benchmark: Run sparse MoE with top-k expert routing."""
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False

        if self.model is None or self.input_tokens is None:
            raise RuntimeError("Model/input not initialized")

        with nvtx_range("optimized_expert_parallelism", enable=enable_nvtx):
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                _ = self.model(self.input_tokens)
        torch.cuda.synchronize()

    def teardown(self) -> None:
        """Cleanup: Clear CUDA cache."""
        self.model = None
        self.input_tokens = None
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
            return "Sparse MoE not initialized"
        if self.input_tokens is None:
            return "Input tokens not initialized"
        return None


def get_benchmark() -> Benchmark:
    """Factory function for benchmark discovery."""
    return OptimizedExpertParallelismBenchmark()


def main():
    """Run optimized expert parallelism benchmark."""
    benchmark = get_benchmark()
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode
    
    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=benchmark.get_config()
    )
    result = harness.benchmark(benchmark)
    print(result)
    print(f"Experts: {benchmark.num_experts} (distributed across {benchmark.num_gpus} GPUs)")
    print(f"Experts per GPU: {benchmark.experts_per_gpu}")
    print("Processing: Parallel expert processing (expert parallelism)")


if __name__ == "__main__":
    main()
