"""Shared workload configuration for Chapter 18 distributed/parallel benchmarks."""

from __future__ import annotations

import os
from dataclasses import dataclass

SMOKE_ENV_FLAG = "BENCHMARK_SMOKE_TEST"


def is_smoke_test() -> bool:
    """Return True when the harness is running in smoke-test mode."""
    return os.environ.get(SMOKE_ENV_FLAG) == "1"


@dataclass(frozen=True)
class Chapter18Workload:
    """Canonical workload sizes for large-scale LLM infrastructure demos."""

    attention_hidden_dim: int = 2048
    attention_num_heads: int = 16
    attention_batch_size: int = 8
    attention_seq_len: int = 2048
    decode_seq_len: int = 256
    micro_batches: int = 8

    pipeline_stages: int = 4
    pipeline_micro_batches: int = 12

    tensor_parallel_shards: int = 4
    distributed_ranks: int = 4
    distributed_global_batch: int = 128

    roofline_matmul_size: int = 4096
    roofline_tile: int = 256

    shared_feature_maps: int = 96
    shared_spatial: int = 192
    shared_kernel_size: int = 5

    smoke_seq_len: int = 512
    smoke_decode_seq_len: int = 64
    smoke_micro_batches: int = 3
    smoke_pipeline_micro_batches: int = 5

    def seq_len(self, smoke: bool = False) -> int:
        return self.smoke_seq_len if smoke else self.attention_seq_len

    def decode_len(self, smoke: bool = False) -> int:
        return self.smoke_decode_seq_len if smoke else self.decode_seq_len

    def micro_batches_for_mode(self, smoke: bool = False) -> int:
        return self.smoke_micro_batches if smoke else self.micro_batches

    def pipeline_micro_batches_for_mode(self, smoke: bool = False) -> int:
        return self.smoke_pipeline_micro_batches if smoke else self.pipeline_micro_batches


WORKLOAD = Chapter18Workload()
