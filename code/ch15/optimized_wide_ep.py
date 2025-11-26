"""optimized_wide_ep.py - Wide expert-parallel simulation for NVL72 story.

This keeps things single-GPU but adds a widening factor knob to illustrate
how spreading experts could reduce per-device load in the real deployment.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class WideExpertMoE(nn.Module):
    """Model that simulates additional experts via grouping."""

    def __init__(self, hidden_dim: int = 1024, num_groups: int = 4, experts_per_group: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_groups = num_groups
        self.experts_per_group = experts_per_group
        self.total_experts = num_groups * experts_per_group
        self.gate = nn.Linear(hidden_dim, self.total_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(self.total_experts)]
        )

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        logits = self.gate(tokens)
        weights = F.softmax(logits, dim=-1)
        top2_w, top2_idx = torch.topk(weights, k=2, dim=-1)

        batch, seq, _ = tokens.shape
        flat_tokens = tokens.view(batch * seq, -1)
        flat_idx = top2_idx.view(batch * seq, 2)
        flat_w = top2_w.view(batch * seq, 2)

        outputs = torch.zeros_like(flat_tokens)
        for expert_id in torch.unique(flat_idx):
            mask = flat_idx[:, 0] == expert_id
            if mask.any():
                outputs[mask] += self.experts[int(expert_id)](flat_tokens[mask]) * flat_w[mask, 0:1]
            mask = flat_idx[:, 1] == expert_id
            if mask.any():
                outputs[mask] += self.experts[int(expert_id)](flat_tokens[mask]) * flat_w[mask, 1:2]
        return outputs.view(batch, seq, -1)


class WideExpertParallelBenchmark(BaseBenchmark):
    """Optimized expert-parallel simulation with widening factor."""

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[WideExpertMoE] = None
        self.inputs: Optional[torch.Tensor] = None
        self._history: Dict[str, float] = {}
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=1024.0,
        )

    def setup(self) -> None:
        torch.manual_seed(2)
        hidden_dim = 1024
        batch = 64
        seq = 16
        self.model = WideExpertMoE(hidden_dim=hidden_dim, num_groups=6, experts_per_group=4).to(self.device).to(
            torch.bfloat16
        )
        self.inputs = torch.randn(batch, seq, hidden_dim, device=self.device, dtype=torch.bfloat16)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if self.model is None or self.inputs is None:
            raise RuntimeError("SKIPPED: wide EP model not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        start = self._record_start()
        with nvtx_range("wide_ep", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.inputs)
        torch.cuda.synchronize(self.device)
        latency_ms = self._record_stop(start)
        self._history["latency_ms"] = latency_ms
        return {"latency_ms": latency_ms}

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload


    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics using standardized helper."""
        from common.python.benchmark_metrics import compute_inference_metrics
        return compute_inference_metrics(
            ttft_ms=getattr(self, '_ttft_ms', 50.0),
            tpot_ms=getattr(self, '_tpot_ms', 10.0),
            total_tokens=getattr(self, 'total_tokens', 256),
            total_requests=getattr(self, 'total_requests', 1),
            batch_size=getattr(self, 'batch_size', 1),
            max_batch_size=getattr(self, 'max_batch_size', 32),
        )

def get_benchmark() -> BaseBenchmark:
    return WideExpertParallelBenchmark()
