"""Shared workload settings for Chapter 1 inference-style benchmarks.

The original toy workloads were too small to highlight the real differences
between the baseline and optimized implementations.  This module centralizes
the “large decode” configuration we validated for kv-cache so every benchmark
can scale to hundreds of tokens/requests without duplicating constants.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Chapter1Workload:
    """Reusable knobs for autoregressive-style microbenchmarks."""

    hidden_dim: int = 512
    batch_size: int = 4
    tokens_per_step: int = 4
    decode_steps: int = 256
    speculative_chunk: int = 8
    request_pool: int = 256
    data_parallel_chunk: int = 32
    prefill_chunks: int = 8
    microbatch_size: int = 32
    performance_microbatches: int = 64
    performance_hidden_dim: int = 8192

    @property
    def total_requests(self) -> int:
        return self.request_pool

    @property
    def total_tokens(self) -> int:
        return self.decode_steps * self.tokens_per_step * self.batch_size


WORKLOAD = Chapter1Workload()
