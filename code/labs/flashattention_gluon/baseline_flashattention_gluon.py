"""Baseline FlashAttention lab: unfused attention (explicit softmax path)."""

from __future__ import annotations

from typing import Optional

import torch

from common.python.benchmark_harness import BaseBenchmark, BenchmarkConfig, WorkloadMetadata
from labs.flashattention_gluon.flashattention_gluon_common import (
    FlashAttentionInputs,
    build_flashattention_inputs,
)


class BaselineFlashAttentionGluonBenchmark(BaseBenchmark):
    """Naive attention: QK^T -> softmax -> matmul V (no fusion, no warp specialization)."""

    def __init__(self) -> None:
        super().__init__()
        self.batch = 2
        self.seq_len = 1024  # modest size to keep baseline cost reasonable
        self.heads = 8
        self.head_dim = 64
        self.dtype = torch.float16
        self.inputs: Optional[FlashAttentionInputs] = None
        tokens = self.batch * self.seq_len
        self._workload = WorkloadMetadata(
            requests_per_iteration=float(self.batch),
            tokens_per_iteration=float(tokens),
        )

    def setup(self) -> None:
        torch.manual_seed(0)
        self.inputs = build_flashattention_inputs(
            batch=self.batch,
            seq_len=self.seq_len,
            heads=self.heads,
            head_dim=self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )
        self._synchronize()

    def benchmark_fn(self) -> None:
        if self.inputs is None:
            raise RuntimeError("FlashAttention inputs are not initialized")

        with torch.inference_mode():
            with self._nvtx_range("flashattention_baseline_unfused"):
                q = self.inputs.q
                k = self.inputs.k
                v = self.inputs.v
                scale = (self.head_dim) ** -0.5
                scores = torch.matmul(q, k.transpose(-1, -2)) * scale
                probs = torch.softmax(scores, dim=-1)
                _ = torch.matmul(probs, v)
        self._synchronize()

    def teardown(self) -> None:
        self.inputs = None
        super().teardown()

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(iterations=6, warmup=5)

    def get_workload_metadata(self) -> Optional[WorkloadMetadata]:
        return self._workload

    def get_custom_metrics(self) -> Optional[dict]:
        """Return domain-specific metrics for performance analysis."""
        # Basic metrics - override in subclass for domain-specific values
        return {
            "flashattention_gluon.workload_size": float(getattr(self, 'batch_size', 0)),
        }

    def validate_result(self) -> Optional[str]:
        if self.inputs is None:
            return "FlashAttention inputs are not initialized"
        return None


def get_benchmark() -> BaseBenchmark:
    return BaselineFlashAttentionGluonBenchmark()
