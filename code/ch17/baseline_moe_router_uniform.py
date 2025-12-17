"""baseline_moe_router_uniform.py - Uniform MoE routing with remote-transfer cost (Ch17).

Pairs with: optimized_moe_router_uniform_topology.py

Design choice (for benchmarkability):
- All experts share the same weights (a single shared expert module).
- Routing decisions therefore do NOT change the final output tensor semantics.
- The measured speedup comes from improved locality (less remote transfer), not
  changing the math.

Baseline behavior:
- Routes tokens uniformly across all islands (topology-agnostic).
- Simulates cross-island expert-parallel communication by gathering "remote"
  tokens and performing device-to-device copies.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

repo_root = Path(__file__).parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.optimization.moe_inference import ExpertMLP


def _pseudo_uniform_expert_ids(token_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    if token_ids.dtype != torch.int64:
        token_ids = token_ids.to(torch.int64)
    return ((token_ids * 1103515245 + 12345) % int(num_experts)).to(torch.int64)


class BaselineMoERouterUniformBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: topology-agnostic routing with high remote-transfer fraction."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024
        self.ffn_size = 4096
        self.num_islands = 4
        self.experts_per_island = 16
        self.num_experts = self.num_islands * self.experts_per_island
        self.batch = 128
        self.seq = 32
        self.dtype = torch.bfloat16

        tokens = self.batch * self.seq
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

        self.expert: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.expert_ids: Optional[torch.Tensor] = None
        self.local_island: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_probe: Optional[torch.Tensor] = None
        self._verify_meta: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for MoE router benchmark")

        if self.num_experts <= 0:
            raise ValueError("num_experts must be positive")
        if self.experts_per_island <= 0:
            raise ValueError("experts_per_island must be positive")
        if self.num_experts != self.num_islands * self.experts_per_island:
            raise ValueError("num_experts must equal num_islands * experts_per_island")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.expert = ExpertMLP(self.hidden_size, self.ffn_size, device=self.device, dtype=self.dtype).eval()
        self.inputs = torch.randn(self.batch, self.seq, self.hidden_size, device=self.device, dtype=self.dtype)

        token_ids = torch.arange(self.batch * self.seq, device=self.device, dtype=torch.int64)
        self.local_island = (token_ids % int(self.num_islands)).to(torch.int64).view(self.batch, self.seq)
        self.expert_ids = _pseudo_uniform_expert_ids(token_ids, self.num_experts).view(self.batch, self.seq)

        self._verify_probe = self.inputs[:1, :1, :256].detach().cpu()
        self._verify_meta = torch.tensor(
            [int(self.num_islands), int(self.experts_per_island), int(self.num_experts)],
            dtype=torch.int64,
        )

        for _ in range(3):
            with torch.no_grad():
                _ = self.expert(self.inputs.view(-1, self.hidden_size))
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.expert is None or self.inputs is None or self.expert_ids is None or self.local_island is None:
            raise RuntimeError("setup() must run before benchmark_fn()")

        flat = self.inputs.view(-1, self.hidden_size)
        expert_ids_flat = self.expert_ids.reshape(-1)
        local_island_flat = self.local_island.reshape(-1)
        dest_island = torch.div(expert_ids_flat, self.experts_per_island, rounding_mode="floor")

        with self._nvtx_range("baseline_moe_router_uniform"):
            with torch.no_grad():
                remote_mask = dest_island != local_island_flat
                remote_idx = remote_mask.nonzero(as_tuple=False).squeeze(-1)
                if remote_idx.numel() > 0:
                    remote_send = flat.index_select(0, remote_idx)
                    remote_recv = torch.empty_like(remote_send)
                    remote_recv.copy_(remote_send)
                    remote_back = torch.empty_like(remote_recv)
                    remote_back.copy_(remote_recv)

                out_flat = self.expert(flat)
                self.output = out_flat.view(self.batch, self.seq, self.hidden_size)

        self._synchronize()

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_probe is None or self._verify_meta is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[:2, :2, :256].detach().cpu().float().clone()
        param_count = sum(p.numel() for p in self.expert.parameters()) if self.expert is not None else 0
        self._set_verification_payload(
            inputs={"probe": self._verify_probe, "topology": self._verify_meta},
            output=output_slice,
            batch_size=int(self.batch),
            parameter_count=int(param_count),
            precision_flags={
                "fp16": False,
                "bf16": True,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.expert = None
        self.inputs = None
        self.expert_ids = None
        self.local_island = None
        self.output = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=10)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineMoERouterUniformBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)

