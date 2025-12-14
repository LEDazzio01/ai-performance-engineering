"""Baseline: sequential autoregressive decoding using only the target model.

Chapter 15 covers speculative decoding as an algorithmic speedup for the decode
phase. This baseline models standard greedy decoding: one target-model forward
pass per generated token.
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata

from ch15.speculative_decoding_common import TokenMLP, default_workload, scale_tail_dims_


class BaselineSpeculativeDecodingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline decode loop: target-only, one token per forward pass."""

    def __init__(self) -> None:
        super().__init__()

        self.workload = default_workload(dtype=torch.bfloat16)

        self.target_model: Optional[TokenMLP] = None
        self.input_ids: Optional[torch.Tensor] = None
        self._output_ids: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

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

        # Deterministic starting token.
        self.input_ids = torch.randint(
            0,
            wl.vocab_size,
            (1, 1),
            device=self.device,
            dtype=torch.int64,
        )

        self._output_ids = torch.empty((1, wl.total_tokens + 1), device=self.device, dtype=torch.int64)
        self.output = None
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.target_model is None or self.input_ids is None or self._output_ids is None:
            raise RuntimeError("Benchmark not initialized")

        wl = self.workload
        out = self._output_ids
        out[:, 0] = self.input_ids[:, 0]

        with self._nvtx_range("baseline_speculative_decoding"):
            with torch.no_grad():
                for t in range(wl.total_tokens):
                    logits = self.target_model(out[:, t : t + 1])
                    out[:, t + 1] = logits[:, 0, :].argmax(dim=-1)

        self.output = out
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
        self.input_ids = None
        self._output_ids = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=5, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if self.output.shape[-1] != self.workload.total_tokens + 1:
            return "Unexpected output shape"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineSpeculativeDecodingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
