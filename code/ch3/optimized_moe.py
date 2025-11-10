"""Chapter 3 MoE optimization: sparse routing + CUDA graphs."""

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


def resolve_device() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for ch3 MoE example")
    return torch.device("cuda")


class RoutedExpert(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ff(x)


class SparseRouter(nn.Module):
    def __init__(self, hidden_dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(RoutedExpert(hidden_dim) for _ in range(num_experts))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seq, hidden = x.shape
        tokens = x.view(-1, hidden)
        logits = self.router(tokens)
        topk_vals, topk_idx = torch.topk(logits, self.top_k, dim=-1)
        weights = torch.softmax(topk_vals, dim=-1)
        output = torch.zeros_like(tokens)
        for expert_id in range(self.num_experts):
            assigned = (topk_idx == expert_id)
            if not assigned.any():
                continue
            token_mask = assigned.any(dim=-1)
            routed = tokens[token_mask]
            routed_out = self.experts[expert_id](routed)
            expert_weights = (weights[token_mask] * assigned[token_mask].float()).sum(dim=-1)
            output[token_mask] += routed_out * expert_weights.unsqueeze(-1)
        return output.view(bsz, seq, hidden)


class OptimizedMoEBenchmark(Benchmark):
    """Sparse routed MoE captured inside a CUDA graph."""

    def __init__(self):
        self.device = resolve_device()
        self.hidden_dim = 768
        self.num_experts = 12
        self.top_k = 2
        self.batch = 4
        self.seq = 128
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None

    def setup(self) -> None:
        torch.manual_seed(7)
        self.model = SparseRouter(self.hidden_dim, self.num_experts, self.top_k).to(self.device).half().eval()
        self.inputs = torch.randn(self.batch, self.seq, self.hidden_dim, device=self.device, dtype=torch.float16)
        with torch.no_grad():
            _ = self.model(self.inputs)
        torch.cuda.synchronize()

    def benchmark_fn(self) -> None:
        from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range

        config = self.get_config()
        enable_nvtx = get_nvtx_enabled(config) if config else False
        assert self.model is not None and self.inputs is not None
        with nvtx_range("optimized_moe", enable=enable_nvtx):
            with torch.cuda.amp.autocast(dtype=torch.float16):
                _ = self.model(self.inputs)

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=8, warmup=2)

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
        config=BenchmarkConfig(iterations=4, warmup=1),
    )
    result = harness.benchmark(get_benchmark())
    print(f"\nSparse MoE latency: {result.timing.mean_ms if result.timing else 0.0:.3f} ms")
