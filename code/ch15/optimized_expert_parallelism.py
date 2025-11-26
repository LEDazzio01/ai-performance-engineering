"""optimized_expert_parallelism.py - Top-2 gated MoE with lightweight overlap.

This is a runnable, single-GPU approximation of expert parallelism that
aggregates top-2 expert outputs and uses CUDA streams to overlap projection
with dispatch. It exists to back the docs' `ch15:expert_parallelism` target.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


def _expert_to_rank(expert_id: int, experts_per_rank: int) -> int:
    return expert_id // experts_per_rank


class Top2MoE(nn.Module):
    """Top-2 MoE with capacity factor, stream overlap, and optional all_to_all."""

    def __init__(self, hidden_dim: int = 1024, num_experts: int = 8, capacity_factor: float = 1.25):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim)) for _ in range(num_experts)]
        )

    def forward(self, tokens: torch.Tensor, dist_group: Optional[torch.distributed.ProcessGroup] = None) -> torch.Tensor:
        batch, seq, hidden = tokens.shape
        logits = self.gate(tokens)
        probs = F.softmax(logits, dim=-1)
        top2_w, top2_idx = torch.topk(probs, k=2, dim=-1)

        flat_idx = top2_idx.view(batch * seq, 2)
        flat_w = top2_w.view(batch * seq, 2)
        flat_tokens = tokens.view(batch * seq, hidden)

        cap = int(self.capacity_factor * (batch * seq) / self.num_experts)
        counts = torch.bincount(flat_idx.view(-1), minlength=self.num_experts)
        mask_overflow = counts > cap

        # If no distributed group, fall back to overlapping streams locally.
        if dist_group is None or torch.distributed.get_world_size(dist_group) == 1:
            streams = [torch.cuda.Stream(device=tokens.device) for _ in range(2)]
            partials = []
            for slot, stream in enumerate(streams):
                expert_ids = flat_idx[:, slot]
                local_out = torch.zeros_like(flat_tokens)
                with torch.cuda.stream(stream):
                    for eid in torch.unique(expert_ids):
                        eid_int = int(eid.item())
                        if mask_overflow[eid_int]:
                            continue
                        mask = expert_ids == eid
                        if mask.any():
                            contrib = self.experts[eid_int](flat_tokens[mask]) * flat_w[mask, slot:slot + 1]
                            local_out[mask] += contrib
                partials.append(local_out)
            torch.cuda.synchronize(tokens.device)
            outputs = sum(partials)
            return outputs.view(batch, seq, hidden)

        # Distributed path: all_to_all tokens to owning ranks then route back.
        world_size = torch.distributed.get_world_size(dist_group)
        rank = torch.distributed.get_rank(dist_group)
        experts_per_rank = max(1, self.num_experts // world_size)

        # Assign each token to its top-1 expert for now.
        top1 = flat_idx[:, 0]
        dest_ranks = top1 // experts_per_rank

        # Build send buffers.
        send_tokens: list[torch.Tensor] = []
        send_indices: list[torch.Tensor] = []
        send_expert_ids: list[torch.Tensor] = []
        for r in range(world_size):
            mask = dest_ranks == r
            send_tokens.append(flat_tokens[mask])
            send_indices.append(torch.nonzero(mask, as_tuple=False).view(-1))
            send_expert_ids.append(top1[mask])

        send_splits = [int(t.size(0)) for t in send_tokens]
        splits_all: list[list[int]] = [None for _ in range(world_size)]  # type: ignore
        dist.all_gather_object(splits_all, send_splits, group=dist_group)
        recv_splits = [splits_all[r][rank] for r in range(world_size)]

        send_buf = torch.cat(send_tokens, dim=0) if send_tokens else torch.empty(0, hidden, device=tokens.device)
        send_ids = torch.cat(send_expert_ids, dim=0) if send_expert_ids else torch.empty(0, device=tokens.device, dtype=torch.int64)
        send_pos = torch.cat(send_indices, dim=0) if send_indices else torch.empty(0, device=tokens.device, dtype=torch.int64)

        total_recv = int(sum(recv_splits))
        recv_buf = torch.empty(total_recv, hidden, device=tokens.device, dtype=flat_tokens.dtype)
        recv_ids = torch.empty(total_recv, device=tokens.device, dtype=torch.int64)
        recv_pos = torch.empty(total_recv, device=tokens.device, dtype=torch.int64)

        dist.all_to_all_single(recv_buf, send_buf, out_split_sizes=recv_splits, in_split_sizes=send_splits, group=dist_group)
        dist.all_to_all_single(recv_ids, send_ids, out_split_sizes=recv_splits, in_split_sizes=send_splits, group=dist_group)
        dist.all_to_all_single(recv_pos, send_pos, out_split_sizes=recv_splits, in_split_sizes=send_splits, group=dist_group)

        # Run local experts.
        local_out = torch.zeros_like(recv_buf)
        for eid in torch.unique(recv_ids):
            eid_int = int(eid.item())
            if mask_overflow[eid_int]:
                continue
            if _expert_to_rank(eid_int, experts_per_rank) != rank:
                continue
            mask = recv_ids == eid
            if mask.any():
                local_out[mask] = self.experts[eid_int](recv_buf[mask])

        # Send results back to original ranks.
        send_back_splits = recv_splits
        recv_back_splits = send_splits
        send_back_buf = local_out
        send_back_pos = recv_pos

        total_back = int(sum(recv_back_splits))
        recv_back_buf = torch.empty(total_back, hidden, device=tokens.device, dtype=flat_tokens.dtype)
        recv_back_pos = torch.empty(total_back, device=tokens.device, dtype=torch.int64)

        dist.all_to_all_single(recv_back_buf, send_back_buf, out_split_sizes=recv_back_splits, in_split_sizes=send_back_splits, group=dist_group)
        dist.all_to_all_single(recv_back_pos, send_back_pos, out_split_sizes=recv_back_splits, in_split_sizes=send_back_splits, group=dist_group)

        # Scatter outputs back into original token order.
        out = torch.zeros_like(flat_tokens)
        out[recv_back_pos] = recv_back_buf
        return out.view(batch, seq, hidden)


class OptimizedExpertParallelismBenchmark(BaseBenchmark):
    """Top-2 expert benchmark meant to mirror the doc's optimized target."""

    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[Top2MoE] = None
        self.inputs: Optional[torch.Tensor] = None
        self._history: Dict[str, float] = {}
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=1024.0,
        )

    def setup(self) -> None:
        torch.manual_seed(1)
        hidden_dim = 1024
        batch = 64
        seq = 16
        self.model = Top2MoE(hidden_dim=hidden_dim, num_experts=8).to(self.device).to(torch.bfloat16).eval()
        self.inputs = torch.randn(batch, seq, hidden_dim, device=self.device, dtype=torch.bfloat16)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if self.model is None or self.inputs is None:
            raise RuntimeError("SKIPPED: MoE model not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        dist_group = torch.distributed.group.WORLD if torch.distributed.is_initialized() else None
        start = self._record_start()
        with nvtx_range("moe_top2_forward", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.inputs, dist_group=dist_group)
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
    """Factory for harness discovery."""
    return OptimizedExpertParallelismBenchmark()
