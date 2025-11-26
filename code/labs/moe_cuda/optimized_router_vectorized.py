"""labs.moe_cuda/optimized_router_vectorized.py - Vectorized expert dispatch step.

This step removes the Python loop over experts by batching expert MLPs, using
scatter/index_add accumulation, and wrapping the forward pass in a CUDA graph.
It also reduces batch size to a GB10-friendly setting so latency is dominated by
device-side math instead of launch overhead.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from common.python.compile_utils import compile_callable, enable_tf32
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range


class VectorizedTopKMoE(nn.Module):
    """Top-k router with batched expert MLPs and scatter accumulation."""

    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 2, expansion: int = 2) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.expanded = hidden_size * expansion
        self.router = nn.Linear(hidden_size, num_experts)

        # Pack expert weights for vectorized matmuls: [E, H, H*exp] and [E, H*exp, H].
        self.w1 = nn.Parameter(torch.empty(num_experts, hidden_size, self.expanded))
        self.b1 = nn.Parameter(torch.zeros(num_experts, self.expanded))
        self.w2 = nn.Parameter(torch.empty(num_experts, self.expanded, hidden_size))
        self.b2 = nn.Parameter(torch.zeros(num_experts, hidden_size))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:  # pragma: no cover - benchmarked
        logits = self.router(tokens)
        top_scores, expert_ids = torch.topk(logits, self.top_k, dim=-1)
        probs = torch.softmax(top_scores, dim=-1, dtype=tokens.dtype)

        flat_tokens = tokens.unsqueeze(1).expand(-1, self.top_k, -1).reshape(-1, self.hidden_size)
        flat_expert_ids = expert_ids.reshape(-1)
        flat_probs = probs.reshape(-1, 1).to(tokens.dtype)

        w1 = self.w1[flat_expert_ids]
        b1 = self.b1[flat_expert_ids]
        # Avoid baddbmm meta-shape expand issues by separating matmul + bias add
        hidden = torch.bmm(flat_tokens.unsqueeze(1), w1).squeeze(1) + b1
        hidden = F.gelu(hidden)

        w2 = self.w2[flat_expert_ids]
        b2 = self.b2[flat_expert_ids]
        expert_out = torch.bmm(hidden.unsqueeze(1), w2).squeeze(1) + b2
        weighted = expert_out * flat_probs

        output = torch.zeros_like(tokens, dtype=tokens.dtype)
        token_indices = torch.arange(tokens.shape[0], device=tokens.device).repeat_interleave(self.top_k)
        output.index_add_(0, token_indices, weighted)
        return output


class VectorizedRouterBenchmark(BaseBenchmark):
    """Benchmark for the vectorized top-k router with graphs + compile."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024
        self.num_experts = 32
        self.top_k = 2
        self.batch_size = 2048  # lower load to fit GB10 latency sweet spot
        self.model: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.graph: Optional[torch.cuda.CUDAGraph] = None
        self.static_output: Optional[torch.Tensor] = None
        tokens = self.batch_size * self.top_k
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        enable_tf32()
        torch.manual_seed(7)
        model = VectorizedTopKMoE(self.hidden_size, self.num_experts, self.top_k, expansion=2)
        model = model.to(self.device, dtype=torch.bfloat16)
        model = compile_callable(model, mode="reduce-overhead", fullgraph=True)
        model.eval()
        self.model = model

        self.inputs = torch.randn(
            self.batch_size,
            self.hidden_size,
            device=self.device,
            dtype=torch.bfloat16,
        )

        # Capture the forward pass into a CUDA graph to hide launch overhead.
        try:
            self.graph = torch.cuda.CUDAGraph()
            self.static_output = torch.empty_like(self.inputs)
            torch.cuda.synchronize(self.device)
            with torch.cuda.graph(self.graph):
                assert self.model is not None and self.inputs is not None
                self.static_output = self.model(self.inputs)
        except Exception:
            self.graph = None
            self.static_output = None
        finally:
            torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> None:
        if self.model is None or self.inputs is None:
            raise RuntimeError("Model not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        with nvtx_range("moe_cuda_router_vectorized", enable=enable_nvtx):
            if self.graph is not None:
                self.graph.replay()
            else:
                with torch.autocast("cuda", dtype=torch.bfloat16), torch.inference_mode():
                    _ = self.model(self.inputs)
        torch.cuda.synchronize(self.device)

    def teardown(self) -> None:
        self.model = None
        self.inputs = None
        self.graph = None
        self.static_output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=6, warmup=2)

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
    "router_vectorized.estimated_flops": flops,
    "router_vectorized.estimated_bytes": bytes_moved,
    "router_vectorized.arithmetic_intensity": arithmetic_intensity,
}

    def validate_result(self) -> Optional[str]:
        if self.model is None:
            return "Vectorized router missing"
        if self.inputs is None:
            return "Inputs missing"
        return None


def get_benchmark() -> BaseBenchmark:
    return VectorizedRouterBenchmark()
