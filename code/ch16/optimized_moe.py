"""optimized_moe.py - Sparse MoE routing benchmark."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import torch
import torch.nn as nn

from common.python.benchmark_harness import Benchmark, BenchmarkConfig
from ch16.moe_workload import (
    resolve_moe_workload,
    MOE_HIDDEN_DIM,
    MOE_NUM_EXPERTS,
)


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch16")
    return torch.device("cuda")


class SparseMoELayer(nn.Module):
    """MoE layer that only executes top-k experts per token using vectorized dispatch."""

    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, num_experts)
        self.experts = nn.ModuleList([ExpertBlock(hidden_dim) for _ in range(num_experts)])

    def _prepare_dispatch(self, router_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        topk_scores, topk_indices = torch.topk(router_logits, self.top_k, dim=-1)
        probs = torch.softmax(topk_scores, dim=-1)
        expert_ids = topk_indices.reshape(-1)
        weights = probs.reshape(-1)
        token_indices = torch.arange(router_logits.shape[0], device=router_logits.device).repeat_interleave(
            self.top_k
        )
        sort_order = torch.argsort(expert_ids)
        return expert_ids[sort_order], weights[sort_order], token_indices[sort_order]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, hidden = x.shape
        tokens = batch * seq_len
        flat_tokens = x.reshape(tokens, hidden)
        router_logits = self.router(flat_tokens)
        expert_ids, weights, token_indices = self._prepare_dispatch(router_logits)

        output = torch.zeros_like(flat_tokens)
        unique_ids, counts = torch.unique_consecutive(expert_ids, return_counts=True)
        start = 0
        for expert_id, count in zip(unique_ids.tolist(), counts.tolist()):
            selected = token_indices[start : start + count]
            dispatch_input = flat_tokens.index_select(0, selected)
            expert_out = self.experts[expert_id](dispatch_input)
            dispatch_weights = weights[start : start + count].unsqueeze(-1)
            output.index_add_(0, selected, expert_out * dispatch_weights)
            start += count
        return output.view(batch, seq_len, hidden)


class ExpertBlock(nn.Module):
    """Simple feed-forward expert block."""

    def __init__(self, hidden_dim: int, expansion: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * expansion),
            nn.ReLU(),
            nn.Linear(hidden_dim * expansion, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class OptimizedMoEBenchmark(Benchmark):
    """Sparse MoE optimized path using top-k routing."""

    def __init__(self):
        self.device = resolve_device()
        workload = resolve_moe_workload()
        self.batch_size = workload.batch_size
        self.seq_len = workload.seq_len
        self.hidden_dim = MOE_HIDDEN_DIM
        self.num_experts = MOE_NUM_EXPERTS
        self.top_k = 2
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.backends.cudnn.benchmark = True
        self.model = SparseMoELayer(self.hidden_dim, self.num_experts, self.top_k).to(self.device).half().eval()
        torch.manual_seed(42)
        self.inputs = torch.randn(
            self.batch_size,
            self.seq_len,
            self.hidden_dim,
            device=self.device,
            dtype=torch.float16,
        )
        with torch.no_grad():
            _ = self.model(self.inputs)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import nvtx_range, get_nvtx_enabled

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.inputs is not None
        with nvtx_range("optimized_moe_sparse", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.inputs)

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            iterations=5,
            warmup=2,
            setup_timeout_seconds=180,
            measurement_timeout_seconds=180,
        )

    def validate_result(self) -> Optional[str]:
        if self.model is None or self.inputs is None:
            return "Model/input not initialized"
        return None


def get_benchmark() -> Benchmark:
    return OptimizedMoEBenchmark()


if __name__ == "__main__":
    from common.python.benchmark_harness import BenchmarkHarness, BenchmarkMode

    harness = BenchmarkHarness(
        mode=BenchmarkMode.CUSTOM,
        config=BenchmarkConfig(iterations=20, warmup=5),
    )
    result = harness.benchmark(get_benchmark())
    timing = result.timing.mean_ms if result.timing else 0.0
    print(f"\nOptimized Sparse MoE: {timing:.3f} ms")
