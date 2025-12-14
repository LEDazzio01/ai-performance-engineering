"""Optimized: speculative decoding with draft proposals + batched verification.

This benchmark models the core speculative-decoding speedup:
  1) Use a small draft model to propose K tokens.
  2) Verify those K tokens in a single target-model forward pass (batch on K).
  3) Accept matching draft tokens; on mismatch, fall back to the target token.

The final generated token sequence must match the baseline greedy decode.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

from ch15.speculative_decoding_common import (
    TokenMLP,
    build_draft_from_target,
    default_workload,
    scale_tail_dims_,
)


class OptimizedSpeculativeDecodingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Spec decode loop: draft + batched target verification."""

    def __init__(self) -> None:
        super().__init__()

        self.workload = default_workload(dtype=torch.bfloat16)

        self.target_model: Optional[TokenMLP] = None
        self.draft_model: Optional[TokenMLP] = None
        self.input_ids: Optional[torch.Tensor] = None
        self._output_ids: Optional[torch.Tensor] = None
        self._draft_ids: Optional[torch.Tensor] = None
        self._verify_prev: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

        self._metrics: Dict[str, float] = {}

        tokens = self.workload.total_tokens * 1.0
        self._workload = WorkloadMetadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=1.0,
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        wl = self.workload
        self.target_model = TokenMLP(
            vocab_size=wl.vocab_size,
            hidden_size=wl.target_hidden,
            num_layers=wl.target_layers,
            device=self.device,
            dtype=wl.dtype,
        ).eval()
        scale_tail_dims_(self.target_model, wl.draft_hidden, wl.tail_scale)

        # Deterministic starting token. Must be created BEFORE draft init so it
        # matches the baseline.
        self.input_ids = torch.randint(
            0,
            wl.vocab_size,
            (1, 1),
            device=self.device,
            dtype=torch.int64,
        )

        self._output_ids = torch.empty((1, wl.total_tokens + 1), device=self.device, dtype=torch.int64)
        self._draft_ids = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.int64)
        self._verify_prev = torch.empty((1, wl.speculative_k), device=self.device, dtype=torch.int64)

        self.draft_model = build_draft_from_target(self.target_model, wl.draft_hidden)
        self.output = None
        self._metrics = {}
        self._synchronize()

    def benchmark_fn(self) -> None:
        if (
            self.target_model is None
            or self.draft_model is None
            or self.input_ids is None
            or self._output_ids is None
            or self._draft_ids is None
            or self._verify_prev is None
        ):
            raise RuntimeError("Benchmark not initialized")

        wl = self.workload
        out = self._output_ids
        out[:, 0] = self.input_ids[:, 0]

        draft_tokens = 0
        accepted_draft = 0
        rounds = 0

        with self._nvtx_range("optimized_speculative_decoding"):
            with torch.no_grad():
                pos = 0
                while pos < wl.total_tokens:
                    rounds += 1
                    remaining = wl.total_tokens - pos
                    k = wl.speculative_k if remaining >= wl.speculative_k else remaining

                    # Draft: propose k tokens sequentially.
                    prev = out[:, pos : pos + 1]
                    for j in range(k):
                        logits_d = self.draft_model(prev)
                        next_d = logits_d[:, 0, :].argmax(dim=-1)
                        self._draft_ids[:, j] = next_d
                        prev = next_d.view(1, 1)

                    draft_tokens += int(k)

                    # Verify: compute target predictions for the k steps in one call.
                    self._verify_prev[:, 0] = out[:, pos]
                    if k > 1:
                        self._verify_prev[:, 1:k] = self._draft_ids[:, : k - 1]

                    logits_t = self.target_model(self._verify_prev[:, :k])
                    target_next = logits_t.argmax(dim=-1)  # [1, k]
                    matches = target_next.eq(self._draft_ids[:, :k])  # [1, k]

                    mismatch = (~matches[0]).nonzero(as_tuple=False)
                    if mismatch.numel() == 0:
                        accept_k = k
                    else:
                        accept_k = int(mismatch[0].item())

                    if accept_k == k:
                        out[:, pos + 1 : pos + k + 1] = self._draft_ids[:, :k]
                        accepted_draft += int(k)
                        pos += k
                    else:
                        if accept_k > 0:
                            out[:, pos + 1 : pos + accept_k + 1] = self._draft_ids[:, :accept_k]
                            accepted_draft += int(accept_k)
                        out[:, pos + accept_k + 1] = target_next[:, accept_k]
                        pos += accept_k + 1

        self.output = out
        self._metrics = {
            "speculative.draft_tokens": float(draft_tokens),
            "speculative.accepted_draft_tokens": float(accepted_draft),
            "speculative.acceptance_rate_pct": (accepted_draft / max(draft_tokens, 1)) * 100.0,
            "speculative.rounds": float(rounds),
        }
        self._synchronize()

    def capture_verification_payload(self) -> None:
        if self.input_ids is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        parameter_count = 0
        if self.target_model is not None:
            parameter_count = sum(p.numel() for p in self.target_model.parameters())
        self._set_verification_payload(
            inputs={"input_ids": self.input_ids},
            output=self.output.float(),
            batch_size=1,
            parameter_count=int(parameter_count),
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.target_model = None
        self.draft_model = None
        self.input_ids = None
        self._output_ids = None
        self._draft_ids = None
        self._verify_prev = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        return dict(self._metrics)

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if self.output.shape[-1] != self.workload.total_tokens + 1:
            return "Unexpected output shape"
        return None


def get_benchmark() -> BaseBenchmark:
    return OptimizedSpeculativeDecodingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
