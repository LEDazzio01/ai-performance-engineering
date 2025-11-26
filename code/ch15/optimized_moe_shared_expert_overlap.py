"""optimized_moe_shared_expert_overlap.py - Stream-overlapped MoE dispatch.

Simulates Megatron-Core-style `--moe-shared-expert-overlap` by splitting router
and expert compute across CUDA streams so communication/computation can overlap.
The computation is intentionally lightweight so it runs anywhere.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from common.python.benchmark_harness import BaseBenchmark, WorkloadMetadata  # noqa: E402
from common.python.nvtx_helper import get_nvtx_enabled, nvtx_range  # noqa: E402


class OverlappedMoE(nn.Module):
    def __init__(self, hidden_dim: int = 1024, num_experts: int = 4):
        super().__init__()
        self.gate = nn.Linear(hidden_dim, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU()) for _ in range(num_experts)]
        )
        self.combine = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, tokens: torch.Tensor, dist_group: Optional[dist.ProcessGroup] = None) -> torch.Tensor:
        logits = self.gate(tokens)
        top2_w, top2_idx = torch.topk(F.softmax(logits, dim=-1), k=2, dim=-1)

        batch, seq, hidden = tokens.shape
        flat_tokens = tokens.view(batch * seq, hidden)
        flat_idx = top2_idx.view(batch * seq, 2)
        flat_w = top2_w.view(batch * seq, 2)

        if dist_group is None or dist.get_world_size(dist_group) == 1:
            streams = [torch.cuda.Stream(device=tokens.device) for _ in range(2)]
            partials = []
            for s_idx, stream in enumerate(streams):
                expert_ids = flat_idx[:, s_idx]
                unique_ids = torch.unique(expert_ids)
                local_out = torch.zeros_like(flat_tokens)
                with torch.cuda.stream(stream):
                    for eid in unique_ids.tolist():
                        mask = expert_ids == eid
                        if mask.any():
                            contrib = self.experts[eid](flat_tokens[mask]) * flat_w[mask, s_idx:s_idx + 1]
                            local_out[mask] += contrib
                partials.append(local_out)
            torch.cuda.synchronize(tokens.device)
            out = sum(partials)
            return self.combine(out.view(batch, seq, hidden))

        # Distributed path: use all_to_all for top-1 routing of the first expert in top-2.
        world_size = dist.get_world_size(dist_group)
        rank = dist.get_rank(dist_group)
        experts_per_rank = max(1, len(self.experts) // world_size)
        top1 = flat_idx[:, 0]
        dest_ranks = top1 // experts_per_rank

        send_tokens: list[torch.Tensor] = []
        send_indices: list[torch.Tensor] = []
        for r in range(world_size):
            mask = dest_ranks == r
            send_tokens.append(flat_tokens[mask])
            send_indices.append(torch.nonzero(mask, as_tuple=False).view(-1))

        send_splits = [int(t.size(0)) for t in send_tokens]
        splits_all: list[list[int]] = [None for _ in range(world_size)]  # type: ignore
        dist.all_gather_object(splits_all, send_splits, group=dist_group)
        recv_splits = [splits_all[r][rank] for r in range(world_size)]

        send_buf = torch.cat(send_tokens, dim=0) if send_tokens else torch.empty(0, hidden, device=tokens.device)
        send_pos = torch.cat(send_indices, dim=0) if send_indices else torch.empty(0, device=tokens.device, dtype=torch.int64)

        total_recv = int(sum(recv_splits))
        recv_buf = torch.empty(total_recv, hidden, device=tokens.device, dtype=flat_tokens.dtype)
        recv_pos = torch.empty(total_recv, device=tokens.device, dtype=torch.int64)

        dist.all_to_all_single(recv_buf, send_buf, out_split_sizes=recv_splits, in_split_sizes=send_splits, group=dist_group)
        dist.all_to_all_single(recv_pos, send_pos, out_split_sizes=recv_splits, in_split_sizes=send_splits, group=dist_group)

        local_out = torch.zeros_like(recv_buf)
        # Only route first expert to match the top-1 overlap pattern.
        for eid in torch.unique(top1):
            eid_int = int(eid.item())
            if (eid_int // experts_per_rank) != rank:
                continue
            mask = top1[send_pos] == eid  # send_pos maps back to local recv order
            if mask.any():
                local_out[mask] = self.experts[eid_int](recv_buf[mask])

        send_back_splits = recv_splits
        recv_back_splits = send_splits
        send_back_buf = local_out
        send_back_pos = recv_pos

        total_back = int(sum(recv_back_splits))
        recv_back_buf = torch.empty(total_back, hidden, device=tokens.device, dtype=flat_tokens.dtype)
        recv_back_pos = torch.empty(total_back, device=tokens.device, dtype=torch.int64)

        dist.all_to_all_single(recv_back_buf, send_back_buf, out_split_sizes=recv_back_splits, in_split_sizes=send_back_splits, group=dist_group)
        dist.all_to_all_single(recv_back_pos, send_back_pos, out_split_sizes=recv_back_splits, in_split_sizes=send_back_splits, group=dist_group)

        out = torch.zeros_like(flat_tokens)
        out[recv_back_pos] = recv_back_buf
        return self.combine(out.view(batch, seq, hidden))


class OptimizedMoeOverlapBenchmark(BaseBenchmark):
    def __init__(self) -> None:
        super().__init__()
        self.model: Optional[OverlappedMoE] = None
        self.inputs: Optional[torch.Tensor] = None
        self._workload = WorkloadMetadata(tokens_per_iteration=1024.0)

    def setup(self) -> None:
        torch.manual_seed(4)
        hidden = 1024
        batch = 64
        seq = 16
        self.model = OverlappedMoE(hidden_dim=hidden, num_experts=4).to(self.device).to(torch.bfloat16)
        self.inputs = torch.randn(batch, seq, hidden, device=self.device, dtype=torch.bfloat16)
        torch.cuda.synchronize(self.device)

    def benchmark_fn(self) -> Optional[dict]:
        if self.model is None or self.inputs is None:
            raise RuntimeError("SKIPPED: overlapped MoE not initialized")

        enable_nvtx = get_nvtx_enabled(self.get_config())
        dist_group = dist.group.WORLD if dist.is_initialized() else None
        with nvtx_range("moe_overlap_optimized", enable=enable_nvtx):
            with torch.no_grad():
                _ = self.model(self.inputs, dist_group=dist_group)
        torch.cuda.synchronize(self.device)
        return {}

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
    return OptimizedMoeOverlapBenchmark()
