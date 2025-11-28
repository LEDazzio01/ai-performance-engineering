"""Shared workload configuration for Chapter 6 ILP microbenchmarks."""

from __future__ import annotations

from dataclasses import dataclass


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

    ilp_iterations: int = 5
    ilp_warmup: int = 5


WORKLOAD = Chapter6Workload()
