"""optimized_moe_overlap_shared_expert.py - Shared-expert MoE with comm overlap (Ch15).

Pairs with: baseline_moe_overlap.py

Semantic contract:
- Both variants compute the same tensor: `shared_expert(x) + routed_expert(x)`.
- The routed expert is shared (identical) across expert ids so routing changes
  do not change output.

Optimization behavior:
- Simulates expert-parallel all-to-all by copying activations into a routed
  buffer on a dedicated CUDA stream.
- Computes the shared expert on the default stream while the copy is in flight
  to model shared-expert overlap with communication.
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

from ch15.verification_payload_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from core.optimization.moe_inference import ExpertMLP
from core.optimization.shared_expert_dispatch import dispatch_shared_expert_active_experts


def _pseudo_uniform_expert_ids(token_ids: torch.Tensor, num_experts: int) -> torch.Tensor:
    if token_ids.dtype != torch.int64:
        token_ids = token_ids.to(torch.int64)
    return ((token_ids * 1103515245 + 12345) % int(num_experts)).to(torch.int64)


class OptimizedMoeOverlapSharedExpertBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Optimized: overlap shared-expert compute with simulated EP communication."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden_size = 1024
        self.ffn_size = 4096
        self.num_experts = 64
        self.batch = 64
        self.seq = 16
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

        self.shared_expert: Optional[nn.Module] = None
        self.routed_expert: Optional[nn.Module] = None
        self.inputs: Optional[torch.Tensor] = None
        self.expert_ids: Optional[torch.Tensor] = None
        self._comm_flat: Optional[torch.Tensor] = None
        self._routed_out_flat: Optional[torch.Tensor] = None
        self._comm_stream: Optional[torch.cuda.Stream] = None
        self.output: Optional[torch.Tensor] = None
        self._verify_probe: Optional[torch.Tensor] = None
        self._verify_meta: Optional[torch.Tensor] = None

    def setup(self) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("SKIPPED: CUDA required for MoE overlap benchmark")

        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.shared_expert = ExpertMLP(self.hidden_size, self.ffn_size, device=self.device, dtype=self.dtype).eval()
        self.routed_expert = ExpertMLP(self.hidden_size, self.ffn_size, device=self.device, dtype=self.dtype).eval()
        self.inputs = torch.randn(self.batch, self.seq, self.hidden_size, device=self.device, dtype=self.dtype)

        token_ids = torch.arange(self.batch * self.seq, device=self.device, dtype=torch.int64)
        self.expert_ids = _pseudo_uniform_expert_ids(token_ids, self.num_experts).view(self.batch, self.seq)
        self._comm_flat = torch.empty(self.batch * self.seq, self.hidden_size, device=self.device, dtype=self.dtype)
        self._routed_out_flat = torch.empty(self.batch * self.seq, self.hidden_size, device=self.device, dtype=self.dtype)
        self._comm_stream = torch.cuda.Stream(device=self.device)

        self._verify_probe = self.inputs[:1, :1, :256].detach().cpu()
        self._verify_meta = torch.zeros(self.num_experts, dtype=torch.int8)

        for _ in range(3):
            with torch.no_grad():
                _ = self.shared_expert(self.inputs.view(-1, self.hidden_size))
                _ = self.routed_expert(self.inputs.view(-1, self.hidden_size))
        self._synchronize()

    def get_custom_streams(self) -> list[torch.cuda.Stream]:
        if self._comm_stream is None:
            return []
        return [self._comm_stream]

    def benchmark_fn(self) -> None:
        if (
            self.shared_expert is None
            or self.routed_expert is None
            or self.inputs is None
            or self.expert_ids is None
            or self._comm_flat is None
            or self._routed_out_flat is None
            or self._comm_stream is None
        ):
            raise RuntimeError("setup() must run before benchmark_fn()")

        flat = self.inputs.view(-1, self.hidden_size)
        expert_ids_flat = self.expert_ids.reshape(-1)

        with self._nvtx_range("optimized_moe_overlap_shared_expert"):
            with torch.no_grad():
                with torch.cuda.stream(self._comm_stream):
                    self._comm_flat.copy_(flat)

                shared_out = self.shared_expert(flat)

                torch.cuda.current_stream(self.device).wait_stream(self._comm_stream)
                dispatch_shared_expert_active_experts(
                    self._comm_flat,
                    expert_ids_flat,
                    self.routed_expert,
                    out=self._routed_out_flat,
                )
                combined = self._routed_out_flat + shared_out
                self.output = combined.view(self.batch, self.seq, self.hidden_size)

        self._synchronize()

    def capture_verification_payload(self) -> None:
        if self.output is None or self._verify_probe is None or self._verify_meta is None:
            raise RuntimeError("setup() and benchmark_fn() must run before capture_verification_payload()")
        output_slice = self.output[:2, :2, :256].detach().cpu().float().clone()
        param_count = 0
        if self.shared_expert is not None:
            param_count += sum(p.numel() for p in self.shared_expert.parameters())
        if self.routed_expert is not None:
            param_count += sum(p.numel() for p in self.routed_expert.parameters())
        self._set_verification_payload(
            inputs={"probe": self._verify_probe, "expert_meta": self._verify_meta},
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
        self.shared_expert = None
        self.routed_expert = None
        self.inputs = None
        self.expert_ids = None
        self._comm_flat = None
        self._routed_out_flat = None
        self._comm_stream = None
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
    return OptimizedMoeOverlapSharedExpertBenchmark()
