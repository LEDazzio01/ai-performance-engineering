"""Baseline guided decoding - CPU mask construction + H2D transfer every step.

Chapter 15 covers high-performance inference, including constrained (guided)
decoding. A naive implementation often rebuilds token-constraint masks on the
CPU and re-uploads them for every decode step, adding avoidable overhead.

This baseline simulates that naive pattern:
  - Build a boolean mask on CPU each step
  - Copy mask to GPU
  - Apply mask to logits and gather a small slice for verification
"""

from __future__ import annotations

from typing import Optional

import torch

from core.benchmark.verification_mixin import VerificationPayloadMixin
from core.harness.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata


class BaselineGuidedDecodingBenchmark(VerificationPayloadMixin, BaseBenchmark):
    """Baseline: rebuild and upload the constraint mask every decode step."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_size = 16
        self.steps = 64
        self.vocab_size = 32768
        self.allowed_count = 4096
        self.output_slice = 256

        self.logits: Optional[torch.Tensor] = None
        self.allowed_token_ids: Optional[torch.Tensor] = None
        self.output: Optional[torch.Tensor] = None

        tokens = self.batch_size * self.steps
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )
        self.register_workload_metadata(
            requests_per_iteration=float(self.batch_size),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(42)
        torch.cuda.manual_seed_all(42)

        self.logits = torch.randn(self.batch_size, self.vocab_size, device=self.device, dtype=torch.float32)
        # Keep token IDs on CPU to force a CPU-side mask build + upload in the baseline.
        self.allowed_token_ids = torch.randperm(self.vocab_size, device="cpu", dtype=torch.int64)[: self.allowed_count]
        self.output = None
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.logits is None or self.allowed_token_ids is None:
            raise RuntimeError("Benchmark not initialized")
        logits = self.logits
        allowed = self.allowed_token_ids

        with self._nvtx_range("baseline_guided_decoding"):
            # Rebuild + upload every step (naive baseline).
            for _ in range(self.steps):
                mask_cpu = torch.zeros(self.vocab_size, dtype=torch.bool, device="cpu")
                mask_cpu[allowed] = True
                mask_gpu = mask_cpu.to(self.device, non_blocking=False)

                masked = logits.masked_fill(~mask_gpu, float("-inf"))
                slice_ids = allowed[: self.output_slice].to(self.device)
                self.output = masked.index_select(1, slice_ids)

        self._synchronize()
        if self.output is None:
            raise RuntimeError("benchmark_fn() must produce output for verification")

    def capture_verification_payload(self) -> None:
        if self.logits is None or self.allowed_token_ids is None or self.output is None:
            raise RuntimeError("benchmark_fn() must run before capture_verification_payload()")
        self._set_verification_payload(
            inputs={
                "logits": self.logits,
                "allowed_token_ids": self.allowed_token_ids,
            },
            output=self.output,
            batch_size=int(self.batch_size),
            parameter_count=0,
            precision_flags={
                "fp16": False,
                "bf16": False,
                "fp8": False,
                "tf32": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else False,
            },
            output_tolerance=(0.0, 0.0),
        )

    def teardown(self) -> None:
        self.logits = None
        self.allowed_token_ids = None
        self.output = None
        torch.cuda.empty_cache()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=20, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def validate_result(self) -> Optional[str]:
        if self.output is None:
            return "Output not produced"
        if not torch.isfinite(self.output).all():
            return "Output contains non-finite values"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineGuidedDecodingBenchmark()


if __name__ == "__main__":
    from core.harness.benchmark_harness import benchmark_main

    benchmark_main(get_benchmark)
