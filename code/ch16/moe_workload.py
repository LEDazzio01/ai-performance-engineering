"""Shared helpers for Chapter 16 MoE benchmarks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch

from common.python.benchmark_utils import warn_benchmark_scaling

MOE_HIDDEN_DIM = 1024
MOE_NUM_EXPERTS = 8
ORIGINAL_BATCH_SIZE = 128
ORIGINAL_SEQ_LEN = 16384


@dataclass(frozen=True)
class MoeWorkload:
    batch_size: int
    seq_len: int


def resolve_moe_workload() -> MoeWorkload:
    """Select a workload size that fits the current GPU but keeps both variants aligned."""
    if torch.cuda.is_available():
        total_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    else:
        total_memory_gb = 0

    if total_memory_gb >= 80:
        batch_size, seq_len = 64, 8192
    elif total_memory_gb >= 40:
        batch_size, seq_len = 32, 4096
    else:
        batch_size, seq_len = 16, 2048

    warn_benchmark_scaling(
        scaling_type="MoE workload size",
        original_values={"batch_size": ORIGINAL_BATCH_SIZE, "seq_len": ORIGINAL_SEQ_LEN},
        scaled_values={"batch_size": batch_size, "seq_len": seq_len},
        impact_description=(
            "Smaller workloads may not fully demonstrate sparse routing benefits; "
            "speedups improve on 80GB+ GPUs."
        ),
        recommendation="Increase batch_size/seq_len manually for production-scale comparisons."
    )

    return MoeWorkload(batch_size=batch_size, seq_len=seq_len)
