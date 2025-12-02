"""Shared workload configuration for ch13 KV cache benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Sequence, Tuple

import torch


@dataclass(frozen=True)
class KVCacheWorkload:
    """Canonical KV cache benchmark settings used by baseline & optimized paths."""

    # Reduced footprint to avoid allocator fragmentation on large B200 runs while
    # keeping enough work to show >1.05x speedups.
    batch_size: int = 2
    num_layers: int = 4
    num_heads: int = 16
    head_dim: int = 64
    sequence_lengths: Tuple[int, ...] = (512, 768, 1024)
    dtype: torch.dtype = torch.float16
    page_size: int = 256
    block_size: int = 128

    @property
    def hidden_dim(self) -> int:
        return self.num_heads * self.head_dim

    @property
    def max_seq_len(self) -> int:
        return max(self.sequence_lengths)

    def lengths(self) -> Tuple[int, ...]:
        """Return the canonical sequence lengths."""
        return self.sequence_lengths


def get_workload() -> KVCacheWorkload:
    """Return the canonical workload settings."""
    return KVCacheWorkload()
