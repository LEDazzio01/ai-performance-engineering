"""Shared workload configuration for Chapter 6 ILP microbenchmarks."""

from __future__ import annotations

import os
from dataclasses import dataclass

SMOKE_ENV_FLAG = "BENCHMARK_SMOKE_TEST"


def is_smoke_test() -> bool:
    """Return True when the harness is running in smoke-test mode."""
    return os.environ.get(SMOKE_ENV_FLAG) == "1"


@dataclass(frozen=True)
class Chapter6Workload:
    """Knobs that scale the ILP-focused benchmarks in Chapter 6."""

    attention_batch: int = 4
    attention_embed_dim: int = 512
    attention_heads: int = 8
    attention_tokens: int = 2048
    attention_chunk_tokens: int = 4

    distributed_elements: int = 8_388_608
    distributed_micro_chunks: int = 64
    distributed_streams: int = 4

    warp_elements: int = 12_582_912
    warp_branch_iterations: int = 32

    quantization_elements: int = 16_777_216

    smoke_attention_tokens: int = 384
    smoke_attention_chunk_tokens: int = 4
    smoke_distributed_elements: int = 4_194_304
    smoke_distributed_micro_chunks: int = 128
    smoke_warp_elements: int = 3_145_728
    smoke_quantization_elements: int = 4_194_304

    ilp_iterations: int = 5
    ilp_warmup: int = 1

    def attention_tokens_for_mode(self, smoke: bool = False) -> int:
        return self.smoke_attention_tokens if smoke else self.attention_tokens

    def attention_chunk_for_mode(self, smoke: bool = False) -> int:
        return self.smoke_attention_chunk_tokens if smoke else self.attention_chunk_tokens

    def distributed_elements_for_mode(self, smoke: bool = False) -> int:
        return self.smoke_distributed_elements if smoke else self.distributed_elements

    def distributed_chunks_for_mode(self, smoke: bool = False) -> int:
        return self.smoke_distributed_micro_chunks if smoke else self.distributed_micro_chunks

    def warp_elements_for_mode(self, smoke: bool = False) -> int:
        return self.smoke_warp_elements if smoke else self.warp_elements

    def quantization_elements_for_mode(self, smoke: bool = False) -> int:
        return self.smoke_quantization_elements if smoke else self.quantization_elements


WORKLOAD = Chapter6Workload()
