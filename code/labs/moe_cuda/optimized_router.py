"""labs.moe_cuda/optimized_router.py - Adaptive top-k MoE router."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import compile_model, enable_tf32
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range


class AdaptiveTopKMoE(nn.Module):
    """Sparse-routing MoE with top-k dispatch and capacity-factor limits."""

    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 2,
        capacity_factor: float = 1.25,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router = nn.Linear(hidden_size, num_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Linear(hidden_size * 2, hidden_size),
            )
            for _ in range(num_experts)
        ])

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover - benchmarked
        logits = self.router(tokens)
        scores, expert_ids = torch.topk(logits, self.top_k, dim=-1)
        probs = torch.softmax(scores, dim=-1)

        flat_expert_ids = expert_ids.reshape(-1)
        flat_token_ids = torch.arange(tokens.shape[0], device=tokens.device).repeat_interleave(self.top_k)
        flat_scores = probs.reshape(-1).to(tokens.dtype)

        sort_order = torch.argsort(flat_expert_ids)
        flat_expert_ids = flat_expert_ids[sort_order]
        flat_token_ids = flat_token_ids[sort_order]
        flat_scores = flat_scores[sort_order]

        capacity = max(1, math.ceil(self.capacity_factor * tokens.shape[0] / self.num_experts))
        output = torch.zeros_like(tokens, dtype=tokens.dtype)

        # Dispatch tokens to experts while enforcing the capacity factor.
        unique_ids, counts = torch.unique_consecutive(flat_expert_ids, return_counts=True)
        start = 0
        for expert_id, count in zip(unique_ids.tolist(), counts.tolist()):
            limit = min(count, capacity)
            shard_tokens = flat_token_ids[start : start + limit]
            shard_scores = flat_scores[start : start + limit].unsqueeze(-1)
            expert_out = self.experts[expert_id](tokens[shard_tokens])
            output.index_add_(0, shard_tokens, expert_out * shard_scores)
            start += count
        return output


class OptimizedRouterTopKBenchmark(BaseBenchmark):
    """Benchmark for the adaptive router."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024
        self.num_experts = 32
        self.top_k = 2
        self.batch_size = 4096
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(self.batch_size * self.top_k),
        )

    def setup(self) -> None:
        enable_tf32()
        torch.manual_seed(0)
        model = AdaptiveTopKMoE(
            self.hidden_size,
            self.num_experts,
            top_k=self.top_k,
            capacity_factor=1.25,
        ).to(self.device, dtype=torch.bfloat16)
        model.eval()
        model = compile_model(model, mode="reduce-overhead")
        self.model = model

        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
        )
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Model not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_cuda_router_topk", enable=enable_nvtx):
            with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                _ = self.model(self.inputs)
        torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=10, warmup=10)  # torch.compile needs warmup

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return roofline analysis metrics."""
        # Estimate problem size for roofline analysis
        n = getattr(self, 'N', 0) or getattr(self, 'hidden_dim', 0) or 4096
        batch = getattr(self, 'batch_size', 1) or getattr(self, 'batch', 1)
        # Simple FLOP estimate for linear layers
        flops = 2.0 * batch * n * n  # Rough estimate
        bytes_moved = batch * n * 4.0  # Input/output bytes
        arithmetic_intensity = flops / max(bytes_moved, 1.0)
        return {
            "router.estimated_flops": flops,
            "router.estimated_bytes": bytes_moved,
            "router.arithmetic_intensity": arithmetic_intensity,
        }

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Adaptive router missing"
        if self.inputs is None:
            return "Inputs missing"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedRouterTopKBenchmark()
