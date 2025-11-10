"""Shared workload configuration for ch13 KV cache benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple

import torch


@dataclass(frozen=True)
class KVCacheWorkload:
    """Canonical KV cache benchmark settings used by baseline & optimized paths."""

    batch_size: int = 4
    num_layers: int = 6
    num_heads: int = 16
    head_dim: int = 64
    sequence_lengths: Tuple[int, ...] = (1024, 1536, 2048)
    smoke_sequence_lengths: Tuple[int, ...] = (1024,)
    dtype: torch.dtype = torch.float16
    page_size: int = 256
    block_size: int = 128

    @property
    def hidden_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def max_seq_len(self) -> int:
        return max(self.sequence_lengths)

    def lengths_for_mode(self, smoke_test: bool = False) -> Tuple[int, ...]:
        """Return sequence lengths for the requested mode."""
        if smoke_test and self.smoke_sequence_lengths:
            return self.smoke_sequence_lengths
        return self.sequence_lengths


def get_workload() -> KVCacheWorkload:
    """Return the canonical workload settings."""
    return KVCacheWorkload()
