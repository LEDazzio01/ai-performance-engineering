"""Shared workload configuration for Chapter 10 Triton benchmarks."""

from __future__ import annotations

import os
from dataclasses import dataclass

SMOKE_ENV_FLAG = "BENCHMARK_SMOKE_TEST"


def is_smoke_test() -> bool:
    """Detect smoke-test mode from the harness."""
    return os.environ.get(SMOKE_ENV_FLAG) == "1"


@dataclass(frozen=True)
class Chapter10Workload:
    """Controls batch/microbatch sizes for Triton and CUDA demos."""

    hidden_dim: int = 2048
    ffn_dim: int = 8192

    triton_batch_size: int = 128
    triton_micro_batches: int = 32

    baseline_micro_batch_size: int = 4
    baseline_micro_batches: int = 128
    optimized_batch_size: int = 512

    roofline_batch_size: int = 256
    roofline_micro_batches: int = 64
    roofline_hidden_dim: int = 4096
    roofline_ffn_dim: int = 8192

    pipeline_micro_batches: int = 64
    pipeline_chunk_tokens: int = 256
    pipeline_hidden_dim: int = 2048

    warp_elements: int = 16_777_216
    warp_branch_iterations: int = 32

    smoke_triton_batch_size: int = 32
    smoke_triton_micro_batches: int = 8
    smoke_baseline_micro_batch_size: int = 4
    smoke_baseline_micro_batches: int = 24
    smoke_optimized_batch_size: int = 256
    smoke_roofline_batch_size: int = 64
    smoke_roofline_micro_batches: int = 8
    smoke_roofline_hidden_dim: int = 2048
    smoke_roofline_ffn_dim: int = 4096
    smoke_pipeline_micro_batches: int = 8
    smoke_pipeline_chunk_tokens: int = 128
    smoke_pipeline_hidden_dim: int = 1536
    smoke_warp_elements: int = 4_194_304

    def triton_batch_for_mode(self, smoke: bool) -> int:
        return self.smoke_triton_batch_size if smoke else self.triton_batch_size

    def triton_micro_batches_for_mode(self, smoke: bool) -> int:
        return self.smoke_triton_micro_batches if smoke else self.triton_micro_batches

    def baseline_micro_batch_for_mode(self, smoke: bool) -> int:
        return self.smoke_baseline_micro_batch_size if smoke else self.baseline_micro_batch_size

    def baseline_micro_batches_for_mode(self, smoke: bool) -> int:
        return self.smoke_baseline_micro_batches if smoke else self.baseline_micro_batches

    def optimized_batch_for_mode(self, smoke: bool) -> int:
        return self.smoke_optimized_batch_size if smoke else self.optimized_batch_size

    def roofline_batch_for_mode(self, smoke: bool) -> int:
        return self.smoke_roofline_batch_size if smoke else self.roofline_batch_size

    def roofline_micro_batches_for_mode(self, smoke: bool) -> int:
        return self.smoke_roofline_micro_batches if smoke else self.roofline_micro_batches

    def roofline_hidden_dim_for_mode(self, smoke: bool) -> int:
        return self.smoke_roofline_hidden_dim if smoke else self.roofline_hidden_dim

    def roofline_ffn_dim_for_mode(self, smoke: bool) -> int:
        return self.smoke_roofline_ffn_dim if smoke else self.roofline_ffn_dim

    def pipeline_micro_batches_for_mode(self, smoke: bool) -> int:
        return self.smoke_pipeline_micro_batches if smoke else self.pipeline_micro_batches

    def pipeline_chunk_tokens_for_mode(self, smoke: bool) -> int:
        return self.smoke_pipeline_chunk_tokens if smoke else self.pipeline_chunk_tokens

    def pipeline_hidden_dim_for_mode(self, smoke: bool) -> int:
        return self.smoke_pipeline_hidden_dim if smoke else self.pipeline_hidden_dim

    def warp_elements_for_mode(self, smoke: bool) -> int:
        return self.smoke_warp_elements if smoke else self.warp_elements

    def warp_branch_iterations_for_mode(self) -> int:
        return self.warp_branch_iterations


WORKLOAD = Chapter10Workload()
